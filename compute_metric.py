import numpy as np
import pickle
from skimage.measure import compare_ssim
import os
import cv2
from termcolor import colored

import argparse
from tqdm import tqdm
import re
import glob
import shutil

from core.utils.image_util import load_image
from core.evaluator import Evaluator

def load_gt_image(gt_path, seg_path, frame_name, cameras, resize_img_scale):

    orig_img = np.array(load_image(gt_path))
    alpha_mask = np.array(load_image(seg_path))
    
    # undistort image
    if frame_name in cameras and 'distortions' in cameras[frame_name[:-4]]:
        K = cameras[frame_name]['intrinsics']
        D = cameras[frame_name]['distortions']
        orig_img = cv2.undistort(orig_img, K, D)
        alpha_mask = cv2.undistort(alpha_mask, K, D)

    alpha_mask = alpha_mask / 255.
    orig_img = orig_img * alpha_mask
    if resize_img_scale != 1.:
        orig_img = cv2.resize(orig_img, None, 
                            fx=resize_img_scale,
                            fy=resize_img_scale,
                            interpolation=cv2.INTER_LANCZOS4)
        alpha_mask = cv2.resize(alpha_mask, None, 
                                fx=resize_img_scale,
                                fy=resize_img_scale,
                                interpolation=cv2.INTER_LINEAR)
                            
    return orig_img, alpha_mask


def cal_metrics(gt_dir, pred_dir, epoch='latest', mode='eval_novel_view', resize_img_scale=0.5):
    evaluator = Evaluator()

    pred_paths_ = os.path.join(pred_dir, epoch, mode)

    #result_dir = os.path.join( pred_paths_ , 'comparison')
    #if os.path.exists(result_dir):
    #    shutil.rmtree(result_dir)

    pred_paths = glob.glob(os.path.join(pred_paths_, '*'))
    #os.system('mkdir -p {}'.format(result_dir))

    for pred_path in tqdm(pred_paths):
        basename = re.split('_', os.path.basename(pred_path)) 
        sub_idx = basename[0]
        cam_idx = basename[1]
        frame_idx = 'frame_' + basename[-1]
        pred_img = np.array(load_image(pred_path)).astype(np.float32)/255.0

        with open(os.path.join(gt_dir, sub_idx, cam_idx, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)

        gt_path = os.path.join(gt_dir, sub_idx, cam_idx, 'images', frame_idx)
        seg_path = os.path.join(gt_dir, sub_idx, cam_idx, 'masks', frame_idx)
        gt_img, seg_img = load_gt_image(gt_path, seg_path, frame_idx, cameras, resize_img_scale)
        gt_img = (gt_img / 255.).astype('float32')

        evaluator.evaluate(pred_img, gt_img)

    evaluator.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input1', type=str, default='dataset/zju_mocap/')
    parser.add_argument('--input2', type=str, default=None)
    parser.add_argument('--epoch', type=str, default='latest')
    parser.add_argument('--mode', type=str, default='eval_novel_view')

    args = parser.parse_args()

    cal_metrics(gt_dir=args.input1, pred_dir=args.input2, epoch=args.epoch, mode=args.mode)
