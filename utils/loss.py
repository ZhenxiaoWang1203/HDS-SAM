from monai.losses import DiceFocalLoss
from scipy.ndimage import convolve

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VSWS(nn.Module):
    def __init__(self):
        super(VSWS, self).__init__()
        self.dice_focal_loss = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')


    def boundary_loss(self, prev_masks, gt3D):
        prev_masks = prev_masks.float()
        gt3D = gt3D.float()
        prev_masks = F.sigmoid(prev_masks)
        prev_grad = torch.abs(F.avg_pool3d(prev_masks, kernel_size=3, stride=1, padding=1) - prev_masks)
        gt3D_grad = torch.abs(F.avg_pool3d(gt3D, kernel_size=3, stride=1, padding=1) - gt3D)
        return F.mse_loss(prev_grad, gt3D_grad)


    def compute_sphericity(self, gt3D, voxel):  #球形度的倒数
        if voxel == 0:
            return 0

        gt = gt3D[0, 0]
        gt = (gt > 0).astype(np.int8)

        kernel = np.zeros((3, 3, 3), dtype=np.int8)
        kernel[1, 1, 0] = 1
        kernel[1, 1, 2] = 1
        kernel[1, 0, 1] = 1
        kernel[1, 2, 1] = 1
        kernel[0, 1, 1] = 1
        kernel[2, 1, 1] = 1

        neighbor_count = convolve(gt, kernel, mode='constant', cval=0)

        surface_voxels = ((gt == 1) & (neighbor_count < 6))
        surface_area = (6 - neighbor_count[surface_voxels]).sum()

        sphericity = (surface_area) / ((np.pi ** (1/3)) * ((6 * voxel) ** (2/3)) + 1e-6)
        return sphericity


    def forward(self, prev_masks, gt3D):
        total_loss = 0
        batch_size = gt3D.shape[0]

        for i in range(batch_size):
            gt_i = gt3D[i:i + 1]
            mask_i = prev_masks[i:i + 1]
            voxel = (gt_i.to(torch.float32) == 1).sum().item()
            sphericity = self.compute_sphericity(gt_i, voxel)
            w = 4.0 / (np.log10(voxel + 10)) * (1 + 0.1 * sphericity)
            loss = 0.8 * self.dice_focal_loss(mask_i, gt_i) + 0.2 * self.boundary_loss(mask_i, gt_i)
            total_loss += loss * w
        return total_loss / batch_size