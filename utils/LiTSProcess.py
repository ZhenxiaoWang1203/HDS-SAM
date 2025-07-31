from __future__ import print_function, division
import sys

sys.path.extend(['../', '../../', './'])

import nibabel as nib
from commons.utils import *

from glob import glob

TMP_DIR = "./tmp"
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)
MIN_IMG_BOUND = -100.0  # Everything below: Water          -1000 for tumor synthesis
MAX_IMG_BOUND = 200.0  # Everything above corresponds to bones    600 for tumor synthesis


def set_bounds(img, MIN_BOUND, MAX_BOUND):
    image = np.clip(img, MIN_BOUND, MAX_BOUND)
    image = image.astype(np.float32)
    return image


def get_image_paths(dataset='../../medical_data/LiTS/ori_data/volume/'):
    image_paths = []
    files = glob(dataset + '*.nii')
    for name in files:
        image_paths.append(name)
    image_paths = sorted(image_paths)
    return image_paths


def load_one_case(item):
    seg_path = item.replace('volume', 'segmentation')
    scan = nib.load(item)
    image = scan.get_fdata()
    image = set_bounds(image, MIN_IMG_BOUND, MAX_IMG_BOUND)
    image = normalize_scale(image)
    seg = nib.load(seg_path).get_fdata()
    return image, seg


def batch_gen_lits(dataset='../../medical_data/LiTS/ori_data/volume/'):
    image_paths = get_image_paths(dataset)
    rootdir = join('../../medical_data/LiTS/processed/')
    if not os.path.isdir(rootdir):
        os.makedirs(rootdir)
    nn = len(image_paths)
    base_path = image_paths[0]
    base_path = base_path[:base_path.rfind('-') + 1]
    for num in range(nn):
        item = base_path + str(num) + '.nii'
        scan = nib.load(item)
        voxel_dim = np.array(scan.header.structarr["pixdim"][1:4], dtype=np.float32)
        voxel_volume = np.prod(voxel_dim)
        image, seg = load_one_case(item)
        tumor_voxel_count = np.sum(seg == 2)
        tumor_volume = tumor_voxel_count * voxel_volume
        print(tumor_volume)
        if len(np.unique(seg)) < 3:
            print("\n\ntotal unique: ", np.unique(seg))
            continue
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg, superior=20, inferior=20)
        image = image[minx:maxx, miny:maxy, minz: maxz]
        seg = seg[minx:maxx, miny:maxy, minz: maxz]

        image = resize(image, (128, 128, 128), mode='edge',
                          cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        seg = resize(seg, (128, 128, 128), order=0, mode='edge',
                              cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        image = set_bounds(image, MIN_IMG_BOUND, MAX_IMG_BOUND)
        image = normalize_scale(image)

        seg = np.where(seg > 1, 1, 0)
        save_file = np.stack([image, seg])
        sample = {
            'data': save_file,
            'tumor_volume': np.float32(tumor_volume),
        }

        np.save(join(rootdir, 'lits_' + '{0:0>3}'.format(num) + '.npy'), sample)


if __name__ == '__main__':
    batch_gen_lits()
