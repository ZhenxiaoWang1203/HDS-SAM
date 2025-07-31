import numpy as np
import random
import datetime
import logging
import matplotlib.pyplot as plt
import os
from monai.transforms import RandGaussianNoised, RandGibbsNoised, \
    RandGaussianSmoothd, RandFlipd, RandAffined, NormalizeIntensityd, ToTensord, Compose, \
    ScaleIntensityRanged, ScaleIntensityd
from torch.utils.data import random_split, Subset
from torch.utils.tensorboard import SummaryWriter

from utils.loss import VSWS
from utils.FistulaProcess import read_csv

join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam3D import sam_model_registry3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2
from utils.FistulaDataLoader import Union_Dataloader, FistulaDataSet
from utils.data_paths import img_datas

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='./ckpt/sam_med3d_turbo.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='work_dir')

# train
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--allow_partial_weight', action='store_true', default=True)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='unet')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=7.5e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def lr_lambda(epoch, total_epochs):
    return (1 - epoch / total_epochs) ** 0.9


def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model


def get_dataloaders(args):
    fistula_training, fistula_test = read_csv('../medical_data/esophageal/esophageal_fistula.csv')
    train_transforms = Compose(
        [
            RandGaussianNoised(keys='img', prob=0.1),
            RandGibbsNoised(keys='img', prob=0.1),
            RandGaussianSmoothd(keys='img', prob=0.1),

            RandFlipd(keys=['img', 'seg'], prob=0.1),
            RandAffined(
                keys=['img', 'seg'],
                prob=0.1,
                rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
                scale_range=(0.05, 0.05, 0.05)
            ),
            ScaleIntensityRanged(keys=['img'], a_min=0, a_max=0.7, b_min=0, b_max=1, clip=True),
            NormalizeIntensityd(keys=['img'], nonzero=False, allow_missing_keys=True),
            ScaleIntensityd(keys=['img'], minv=0.0, maxv=1.0, allow_missing_keys=True),
            ToTensord(keys=['img', 'seg', 'num', 'cat'])
        ]
    )
    val_transforms = Compose(
        [
            ScaleIntensityRanged(keys=['img'], a_min=0, a_max=0.7, b_min=0, b_max=1, clip=True),
            NormalizeIntensityd(keys=['img'], nonzero=False, allow_missing_keys=True),
            ScaleIntensityd(keys=['img'], minv=0.0, maxv=1.0, allow_missing_keys=True),
            ToTensord(keys=['img', 'seg', 'num', 'cat'])
        ]
    )
    dataset = FistulaDataSet(data=fistula_training, transform=train_transforms,
                             root_dir='../medical_data/esophageal/fistula',
                             text_only=False, input_size=(128, 128, 128))
    train_size = 350
    val_size = 40
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=[train_size, val_size])
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    t_dataset = FistulaDataSet(data=fistula_training, transform=train_transforms,
                               root_dir='../medical_data/esophageal/fistula',
                               text_only=False, input_size=(128, 128, 128))
    v_dataset = FistulaDataSet(data=fistula_training, transform=val_transforms,
                               root_dir='../medical_data/esophageal/fistula',
                               text_only=False, input_size=(128, 128, 128))
    train_dataset = Subset(t_dataset, train_indices)
    val_dataset = Subset(v_dataset, val_indices)

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = False

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataloader = Union_Dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader


class BaseTrainer:
    def __init__(self, model, dataloaders, val_dataloaders, args, writer):

        self.model = model
        self.dataloaders = dataloaders
        self.val_dataloaders = val_dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.val_losses = []
        self.val_dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if (args.resume):
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        self.writer = writer

    def set_loss_fn(self):
        self.seg_loss = VSWS()

    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW([
            {'params': sam_model.image_encoder.parameters()},  # , 'lr': self.args.lr * 0.1},
            {'params': sam_model.prompt_encoder.parameters(), 'lr': self.args.lr * 0.1},
            {'params': sam_model.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
        ], lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                     self.args.step_size,
                                                                     self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        elif self.args.lr_scheduler == 'unet':
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                  lr_lambda=lambda cur_epoch: lr_lambda(cur_epoch + 1,
                                                                                                        200))
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)

        if last_ckpt:
            if (self.args.allow_partial_weight):
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            else:
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size // 4, args.img_size // 4, args.img_size // 4))
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=[points_input, labels_input])
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss

    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()

    def train_epoch(self, epoch, num_clicks, writer):
        epoch_loss = 0
        epoch_iou = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
            # tbar = self.dataloaders
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        epoch_dice = 0
        step = 0
        pred_list = []
        for step, batch in enumerate(tbar):

            my_context = self.model.no_sync if self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(batch['image_patch'].squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                gt3D = batch['image_segment'].to(device).type(torch.long)
                # with amp.autocast():

                image_embedding = sam_model.to(device).image_encoder(image3D)

                self.click_points = []
                self.click_labels = []

                prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, num_clicks=11)

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(prev_masks, gt3D)
                cur_loss = loss.item()

                loss /= self.args.accumulation_steps

                self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss

        epoch_loss /= step + 1
        epoch_dice /= step + 1

        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Dice', epoch_dice, epoch)

        return epoch_loss, epoch_iou, epoch_dice, pred_list

    def eval_epoch(self, epoch, num_clicks, writer):
        self.model.eval()
        sam_model = self.model
        val_loss = 0.0
        val_dice = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloaders):
                image3D = self.norm_transform(batch['image_patch'].squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)  # batch, 1, 128, 128, 128
                gt3D = batch['image_segment'].to(device).type(torch.long)

                image_embedding = sam_model.image_encoder(image3D)

                self.click_points = []
                self.click_labels = []

                prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, num_clicks=11)

                val_loss += loss.item()

                dice_score = self.get_dice_score(prev_masks, gt3D)
                val_dice += dice_score

                total_batches += 1

        avg_val_loss = val_loss / total_batches
        avg_val_dice = val_dice / total_batches

        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/Dice', avg_val_dice, epoch)

        return avg_val_loss, avg_val_dice

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()

    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks, self.writer)
            val_loss, val_dice = self.eval_epoch(epoch, num_clicks, self.writer)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                self.val_losses.append(val_loss)
                self.val_dices.append(val_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                print(f'EPOCH: {epoch}, Val_Loss: {val_loss}')
                print(f'EPOCH: {epoch}, Val_Dice: {val_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}, val_loss: {val_loss}, val_dice: {val_dice}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                # save latest checkpoint
                self.save_checkpoint(
                    epoch,
                    state_dict,
                    describe='latest'
                )

                # save train loss best checkpoint
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='loss_best'
                    )

                # save train dice best checkpoint
                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='dice_best'
                    )

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args,)
        )
    else:
        random.seed(8888)
        np.random.seed(8888)
        torch.manual_seed(8888)
        # Load datasets
        dataloaders, val_dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)

        for param in model.image_encoder.patch_embed.parameters():
            param.requires_grad = False
        model.image_encoder.pos_embed.requires_grad = False
        for param in model.image_encoder.blocks.parameters():
            param.requires_grad = False
        for param in model.image_encoder.neck.parameters():
            param.requires_grad = False

        writer = SummaryWriter(log_dir='./logs')
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, val_dataloaders, args, writer)
        # Train
        trainer.train()


def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(8888 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))

    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
