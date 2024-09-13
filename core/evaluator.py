import numpy as np
from skimage.measure import compare_ssim
import os
import cv2
from termcolor import colored

import torch
from third_parties.lpips import LPIPS

def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.

class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.lpips_model = LPIPS(net='vgg').cuda()

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt):
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        return ssim

    def lpips_metric(self, img_pred, img_gt):
        img_pred = torch.tensor(img_pred[None, ...]).float().cuda()
        img_gt = torch.tensor(img_gt[None, ...]).float().cuda()
        lpips_loss = self.lpips_model(scale_for_lpips(img_pred.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(img_gt.permute(0, 3, 1, 2)))
        lpips_loss = torch.mean(lpips_loss)
        return lpips_loss.item()

    def evaluate(self, rgb_pred, rgb_gt):

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt)
        self.ssim.append(ssim)

        lpips = self.lpips_metric(rgb_pred, rgb_gt)
        self.lpips.append(lpips)

    def summarize(self):
        print('mse: {}'.format(np.mean(self.mse)))
        print('psnr: {}'.format(np.mean(self.psnr)))
        print('ssim: {}'.format(np.mean(self.ssim)))
        print('lpips: {}'.format(np.mean(self.lpips)))
