from glob import glob
from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset


class LiTS2017(Dataset):
    def __init__(self, base_dir=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = sorted(glob(join(self._base_dir, '*.npy')), reverse=False)
        img_names = set()
        for name in self.image_list:
            img_names.add(name)
        print("Total {} samples".format(len(self.image_list)))
        print('Names:', sorted(list(img_names)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        img = np.load(image_name, allow_pickle=True).item()['data']
        tumor_volume = np.load(image_name, allow_pickle=True).item()['tumor_volume']
        img_ori = np.expand_dims(img[0], 0)
        seg = torch.from_numpy(np.expand_dims(img[1], 0))
        sample = {'image': img_ori, 'label': seg.type(torch.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return {
            "image_patch": sample['image'],
            "image_segment": sample['label'],
            "tumor_volume": np.float32(tumor_volume),
            "image_path": image_name,
        }
