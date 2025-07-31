import shutil
import sys
from os.path import join, exists, basename

import numpy as np
import torch
import pandas as pd
from torchvision.utils import make_grid

sys.path.extend(["../../", "../", "./"])
from PIL import Image
import SimpleITK as sitk
import os
from skimage.transform import resize

data_root = "../../medical_data/esophageal"
data_path = "../../medical_data/esophageal/fistula"
csv_data_path = "../../medical_data/esophageal/esophageal_fistula.csv"
MIN_IMG_BOUND = -200  # Everything below: Water  -62   -200
MAX_IMG_BOUND = 300  # Everything above corresponds to bones  238    200
MIN_MSK_BOUND = 0.0  # Everything above corresponds
MAX_MSK_BOUND = 2.0  # Everything above corresponds


def set_bounds(img, MIN_BOUND, MAX_BOUND):
    image = np.clip(img, MIN_BOUND, MAX_BOUND)
    image = image.astype(np.float32)
    return image


def dcm_count(file_list):
    dcm_count = 0
    for file_name in file_list:
        if file_name.split('.')[1] == 'dcm':
            dcm_count += 1
    return dcm_count


def find_scan(path):
    path_list = []  # paths that contain .dcm files
    dcm_counts = []  # num of .dcm files

    subdir_list = [name for name in os.listdir(path)
                   if not name.endswith('-ca')]
    for subdir in subdir_list:
        subdir = os.path.join(path, subdir)
        file_list = os.listdir(subdir)
        print(file_list)
        count = dcm_count(file_list)
        path_list.append(subdir)
        dcm_counts.append(count)

    return path_list


def read_dicom(scan_path):
    reader = sitk.ImageSeriesReader()
    try:
        dicom_names = reader.GetGDCMSeriesFileNames(scan_path)
    except Exception as er:
        print(er)
    print(os.path.basename(scan_path))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    origin = image.GetOrigin()  # x, y, z
    spacing = image.GetSpacing()  # x, y, z
    deps = image_array.shape[0]
    cols = image_array.shape[1]
    rows = image_array.shape[2]
    return image_array, spacing, origin


def read_csv(csv_path):
    fistula_df = pd.read_csv(csv_path,
                             usecols=['name', '训练组a验证组b', 'Fistula', '序号', '***'])
    fistula_df['序号'] = fistula_df['序号'].map(lambda x: '{:g}'.format(x))
    fis_cat = pd.DataFrame(fistula_df, columns=['****'])
    fis_num = pd.DataFrame(fistula_df, columns=['*****'])
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    minus = lambda x: (x - 1) if np.min(x) > 0 else x
    for index, row in fis_num.items():
        fis_num[index] = fis_num[[index]].apply(max_min_scaler)
    np_num = fis_num.values
    for index, col in fis_cat.items():
        fis_cat[index] = fis_cat[[index]].apply(minus)
    cat_total = np.max(fis_cat.values, axis=0) + 1
    np_cat = np.zeros((fis_cat.values.shape[0], np.sum(cat_total))).astype(int)
    for index, row in fis_cat.iterrows():
        tmp = []
        for idx in range(len(row.values)):
            onehot = torch.zeros(cat_total[idx])
            tmp.append(np.array(onehot.scatter_(0, torch.tensor(row.values[idx]), 1).flatten()))
        np_cat[index] = np.concatenate(tmp).astype(int)
    fistula_df_training, fistula_df_test = [], []
    for index, row in fistula_df.iterrows():
        if row['训练组a验证组b'] == 'a':
            fistula_df_training.append({'name': row['name'],
                                        'Fistula': row['Fistula'],
                                        '序号': row['序号'],
                                        'num': np_num[index],
                                        'cat': np_cat[index]})
        else:
            fistula_df_test.append({'name': row['name'],
                                    'Fistula': row['Fistula'],
                                    '序号': row['序号'],
                                    'num': np_num[index],
                                    'cat': np_cat[index]})
    return fistula_df_training, fistula_df_test


def process_dicom_file(new_img_dir='Seg_Processed', truncate=[MIN_IMG_BOUND, MAX_IMG_BOUND],
                       root_dir='../../medical_data/esophageal', resample=False, new_voxel_dim=[1, 1, 1],
                       min_max=False, p_size=[128, 128, 128]):
    new_img_dir = new_img_dir + '_%d' % (truncate[1])
    new_file_dir = join(root_dir, new_img_dir)
    original_files = join(root_dir, 'fistula')
    if resample:
        new_file_dir = new_file_dir + '_Resample'
    if min_max:
        new_file_dir = new_file_dir + '_MM'
    if not exists(new_file_dir):
        os.makedirs(new_file_dir)
    shutil.rmtree(new_file_dir)
    os.makedirs(new_file_dir)
    files = find_scan(original_files)
    files.sort(reverse=True)
    total_slices = 0
    gmaxx, gmaxy, gmaxz = 0, 0, 0
    for ff in files:
        print(ff)
        image_array, spacing, origin = read_dicom(ff)
        image_array_seg, spacing_seg, origin_seg = read_dicom(ff + '-ca')
        print(spacing)
        print(origin_seg)
        print('%s origin value range: ' % (basename(ff)), (np.min(image_array), np.max(image_array)))
        print('seg value range: ', np.unique(image_array_seg))
        print(image_array.shape)
        print('min hu %d, max hu %d' % (np.min(image_array), np.max(image_array)))
        minz, maxz, minx, maxx, miny, maxy = min_max_voi(image_array_seg, superior=1, inferior=1)
        gmaxx = maxx - minx if (maxx - minx) > gmaxx else gmaxx
        gmaxy = maxy - miny if (maxy - miny) > gmaxy else gmaxy
        gmaxz = maxz - minz if (maxz - minz) > gmaxz else gmaxz
        print('gmaxx %.3f, gmaxy %.3f, gmaxz %.3f' % (gmaxx, gmaxy, gmaxz))
        if resample:
            # TODO
            print('TODO')
        truncate_low = truncate[0]
        truncate_high = truncate[1]
        if np.min(image_array) > 0:
            truncate_low = 0
            truncate_high = 255
        print('min hu %.3f, max hu %.3f' % (truncate_low, truncate_high))
        img = set_bounds(image_array, truncate_low, truncate_high)
        img = np.transpose(img, (1, 2, 0))
        midx = (maxx + minx) / 2
        midy = (maxy + miny) / 2
        midz = (maxz + minz) / 2
        if img.shape[-1] > p_size[-1]:
            img_crop = img[int(midx - p_size[0] / 2):int(midx + p_size[0] / 2),
                       int(midy - p_size[1] / 2):int(midy + p_size[1] / 2),
                       int(max(midz, p_size[2] / 2) - p_size[2] / 2):int(
                           min(midz + p_size[2] / 2, img.shape[-1]) + p_size[2] / 2)]
        else:
            img_crop = img[int(midx - p_size[0] / 2):int(midx + p_size[0] / 2),
                       int(midy - p_size[1] / 2):int(midy + p_size[1] / 2), :]
        image_array_seg = np.transpose(image_array_seg, (1, 2, 0))
        if img.shape[-1] > p_size[-1]:
            img_seg_crop = image_array_seg[int(midx - p_size[0] / 2):int(midx + p_size[0] / 2),
                           int(midy - p_size[1] / 2):int(midy + p_size[1] / 2),
                           int(max(midz, p_size[2] / 2) - p_size[2] / 2):int(
                               min(midz + p_size[2] / 2, img.shape[-1]) + p_size[2] / 2)]
        else:
            img_seg_crop = image_array_seg[int(midx - p_size[0] / 2):int(midx + p_size[0] / 2),
                           int(midy - p_size[1] / 2):int(midy + p_size[1] / 2), :]
        img_crop = resize(img_crop, (128, 128, 128),  mode='edge',
                           cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        img_seg_crop = resize(img_seg_crop, (128, 128, 128), order=0, mode='edge',
                                   cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        img_crop = normalize_scale(img_crop)
        final_data = torch.from_numpy(img_crop).unsqueeze(dim=0).unsqueeze(dim=0)
        visual_batch(final_data, './tmp', basename(ff), channel=1, nrow=8)
        final_data = torch.from_numpy(img_seg_crop).unsqueeze(dim=0).unsqueeze(dim=0)
        visual_batch(final_data, './tmp', basename(ff) + '-ca', channel=1, nrow=8)
        print("crop image size: ", img_crop.shape)
        print("crop img_seg_crop size: ", img_seg_crop.shape)
        final = np.stack([img_crop, img_seg_crop])
        np.save(join(new_file_dir, basename(ff) + '.npy'), final)
        generate_txt(img_seg_crop, join(new_file_dir, basename(ff) + '_fistula'))
    print(total_slices)


def min_max_voi(mask, superior=10, inferior=10):
    sp = mask.shape
    tp = np.transpose(np.nonzero(mask))
    minx, miny, minz = np.min(tp, axis=0)
    maxx, maxy, maxz = np.max(tp, axis=0)
    minz = 0 if minz - superior < 0 else minz - superior
    maxz = sp[-1] if maxz + inferior >= sp[-1] else maxz + inferior + 1
    miny = 0 if miny - superior < 0 else miny - superior
    maxy = sp[1] if maxy + inferior >= sp[1] else maxy + inferior + 1
    minx = 0 if minx - superior < 0 else minx - superior
    maxx = sp[0] if maxx + inferior >= sp[0] else maxx + inferior + 1
    return minx, maxx, miny, maxy, minz, maxz


def normalize_scale(img):
    imgs_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    return imgs_normalized


def visual_batch(batch, dir, name, channel=1, nrow=8):
    batch_len = len(batch.size())
    if batch_len == 3:
        image_save = batch.detach().contiguous()
        image_save = image_save.unsqueeze(1)
        grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
    if batch_len == 4:
        if channel == 3:
            image_save = batch.detach().contiguous()
            grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
            save_img = grid.detach().cpu().numpy()
            visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
        else:
            image_save = batch.detach().contiguous()
            image_save = image_save.view(
                (image_save.size(0) * image_save.size(1), image_save.size(2), image_save.size(3)))
            image_save = image_save.unsqueeze(1)
            grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
            save_img = grid.detach().cpu().numpy()
            visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
    if batch_len == 5:
        if channel == 3:
            image_save = batch.transpose(1, 4).contiguous()
            image_save = image_save.view(
                (image_save.size(0) * image_save.size(1), image_save.size(2), image_save.size(3), image_save.size(4)))
            image_save = image_save.permute(0, 3, 1, 2)
            grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
            save_img = grid.detach().cpu().numpy()
            visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
        else:
            image_save = batch.transpose(1, 4).contiguous()
            image_save = image_save.view(
                (image_save.size(0) * image_save.size(1), image_save.size(4), image_save.size(2), image_save.size(3)))
            grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
            save_img = grid.detach().cpu().numpy()
            visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))


def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    save_img = data - np.min(data) / (np.max(data) - np.min(data))
    save_img = np.clip(save_img * 255 + 0.5, 0, 255)
    img = Image.fromarray(save_img.astype(np.uint8))  # the image is already 0-255
    img.save(filename + '.png')
    return img


def generate_txt(msk, save_folder):
    f = open(save_folder + '.txt', 'w')
    index = np.where(msk > 0)
    x = index[0]
    y = index[1]
    z = index[2]
    np.savetxt(f, np.c_[x, y, z], fmt="%d")
    f.write("\n")
    f.close()


if __name__ == '__main__':
    process_dicom_file(new_img_dir='fistula-Seg_Processed',
                       root_dir=data_root, resample=False,
                       new_voxel_dim=[1, 1, 1],
                       min_max=False, p_size=[200, 200, 130])
    fistula_training, fistula_test = read_csv(csv_data_path)
    print(fistula_training)
    print(fistula_test)
    path_list = find_scan(data_path)
    path_list = sorted(path_list)
    for path in path_list:
        read_dicom(path)
