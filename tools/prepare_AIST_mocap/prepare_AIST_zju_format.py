#####################################################################################################################
####################################  extract videos from full AIST++  ##############################################
#####################################################################################################################

import shutil
import os

src_pth = './AIST_plus'
dst_video_pth = './AIST_slc/videos'
dst_img_pth = './AIST_slc/imgs'
cam_pth = './20210308_cameras/cameras'
smpl_pth = './20210308_motions/motions'
if not os.path.exists(dst_video_pth):
    os.makedirs(dst_video_pth, exist_ok=True)
if not os.path.exists(dst_img_pth):
    os.makedirs(dst_img_pth, exist_ok=True)

src_list = []
src_list.append([
'gJS_sFM_c0{}_d01_mJS5_ch06.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gJS_sFM_c0{}_d02_mJS3_ch04.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gJS_sFM_c0{}_d03_mJS5_ch13.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gBR_sFM_c0{}_d04_mBR0_ch01.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gBR_sFM_c0{}_d05_mBR1_ch08.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gBR_sFM_c0{}_d06_mBR2_ch16.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gJB_sFM_c0{}_d07_mJB2_ch03.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gJB_sFM_c0{}_d08_mJB2_ch10.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gJB_sFM_c0{}_d09_mJB1_ch21.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gPO_sFM_c0{}_d10_mPO3_ch07.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gPO_sFM_c0{}_d11_mPO2_ch10.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gPO_sFM_c0{}_d12_mPO1_ch16.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gLO_sFM_c0{}_d13_mLO0_ch01.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gLO_sFM_c0{}_d14_mLO2_ch10.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gLO_sFM_c0{}_d15_mLO0_ch15.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gLH_sFM_c0{}_d16_mLH2_ch03.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gLH_sFM_c0{}_d17_mLH1_ch09.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gLH_sFM_c0{}_d18_mLH1_ch16.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gHO_sFM_c0{}_d19_mHO4_ch05.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gHO_sFM_c0{}_d20_mHO3_ch11.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gHO_sFM_c0{}_d21_mHO3_ch18.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gMH_sFM_c0{}_d22_mMH2_ch03.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gMH_sFM_c0{}_d23_mMH4_ch12.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gMH_sFM_c0{}_d24_mMH1_ch16.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gWA_sFM_c0{}_d25_mWA2_ch03.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gWA_sFM_c0{}_d26_mWA1_ch09.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gWA_sFM_c0{}_d27_mWA4_ch19.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gKR_sFM_c0{}_d28_mKR4_ch05.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gKR_sFM_c0{}_d29_mKR3_ch11.mp4'.format(i) for i in range(1,10)])
src_list.append([
'gKR_sFM_c0{}_d30_mKR5_ch20.mp4'.format(i) for i in range(1,10)])

#for performer_id in range(30): 
#    for p in src_list[performer_id]:
#        shutil.copyfile(os.path.join(src_pth,p), os.path.join(dst_video_pth,p))


#####################################################################################################################
########################################### extract video to imgs ###################################################
#####################################################################################################################

import cv2
from tqdm import tqdm
import re

def extract_video(videoname, dst_img_pth, end):
    start = 200
    base = os.path.basename(videoname).replace('.mp4', '')
    if not os.path.exists(videoname):
        return base
    video = cv2.VideoCapture(videoname)
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    performer_id = re.split('_', base)[3]
    cam_id = re.split('_', base)[2]
    if not os.path.exists( os.path.join(dst_img_pth, performer_id, cam_id ) ):
        os.makedirs(os.path.join(dst_img_pth, performer_id, cam_id ), exist_ok=True)
    for cnt in tqdm(range(totalFrames), desc='{:10s}'.format(os.path.basename(videoname))):
        ret, frame = video.read()
        if cnt < start:continue
        if cnt % 4 != 0:continue
        #if cnt >= end:break
        if not ret:continue
        w_pth =  os.path.join(dst_img_pth, performer_id, cam_id, '{:06d}.jpg'.format(cnt) )
        cv2.imwrite(w_pth, frame)
    video.release()
    return base


#####################################################################################################################
########################################### load camera parameters ##################################################
#####################################################################################################################
# load imgs and SMPL model params, repro for mask

import numpy as np
import json

def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_cams(video_basename, trans_scale):
    cam_mapping_data = np.genfromtxt( os.path.join(cam_pth, 'mapping.txt'), dtype='str')
    cam_mapping = {}
    for i in range(cam_mapping_data.shape[0]):
        cam_mapping[cam_mapping_data[i][0]] = cam_mapping_data[i][1]
    cam_setting = cam_mapping[video_basename] 

    with open(  os.path.join( cam_pth, '{}.json'.format(cam_setting))  ) as j:
        cam_params = json.loads(j.read())

    R_lst = []
    T_lst = []
    K_lst = []
    D_lst = []
    for i in range(9):
        D_lst.append(np.array(cam_params[i]['distortions']).reshape(5,1))
        R = batch_rodrigues(np.array(cam_params[i]['rotation']).reshape(1,3))
        R_lst.append(R[0].reshape(3,3))
        T_lst.append(np.array(cam_params[i]['translation']).reshape(3,1)/trans_scale*1000)
        K_lst.append(np.array(cam_params[i]['matrix']).reshape(3,3))
    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    cams['K'] = K_lst
    cams['D'] = D_lst
    cams['R'] = R_lst
    cams['T'] = T_lst
    return cams


#####################################################################################################################
########################################### load SMPL parameters and proj ###########################################
#####################################################################################################################
# load imgs and SMPL model params, repro for mask
import pickle
from smplx import SMPL
from aist_plusplus.loader import AISTDataset
import trimesh
import glob
import torch

def get_smpl_params(video_basename):
    with open( os.path.join(smpl_pth, video_basename+'.pkl'), 'rb') as f:
        smpl_data = pickle.load(f)
    smpl_poses = smpl_data['smpl_poses'] # (N, 72)
    smpl_trans = smpl_data['smpl_trans'] # (N, 3)
    smpl_scaling = smpl_data['smpl_scaling'] # (1, )

    smpl_dir = './aistplusplus_api/data/smpl/'

    # SMPL forward path
    smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1)
    smpl_model = smpl.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:3]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 3:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
        scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
        )
    new_vertices = smpl_model.vertices.detach().numpy()

    new_params = {}
    new_params['Rh'] = smpl_poses[:,:3].copy()
    new_params['Th'] = smpl_trans
    new_params['poses'] = smpl_poses.copy()
    new_params['poses'][:,:3] = 0.0
    new_params['scaling'] = smpl_scaling

    return new_params, new_vertices

def check_ignore_list():
    ignore_lst = [
        'gBR_sFM_cAll_d05_mBR5_ch14',
        'gBR_sFM_cAll_d06_mBR5_ch19',
        'gBR_sFM_cAll_d04_mBR4_ch07',
        'gBR_sFM_cAll_d05_mBR4_ch13',
        'gBR_sBM_cAll_d04_mBR0_ch08',
        'gBR_sBM_cAll_d04_mBR0_ch07',
        'gBR_sBM_cAll_d04_mBR0_ch10',
        'gBR_sBM_cAll_d05_mBR0_ch07',
        'gJB_sBM_cAll_d07_mJB2_ch06',
        'gJB_sBM_cAll_d07_mJB3_ch01',
        'gJB_sBM_cAll_d07_mJB3_ch05',
        'gJB_sBM_cAll_d08_mJB0_ch07',
        'gJB_sBM_cAll_d08_mJB0_ch09',
        'gJB_sBM_cAll_d08_mJB1_ch09',
        'gJB_sFM_cAll_d08_mJB3_ch11',
        'gJB_sBM_cAll_d08_mJB5_ch07',
        'gJB_sBM_cAll_d09_mJB2_ch07',
        'gJB_sBM_cAll_d09_mJB3_ch07',
        'gJB_sBM_cAll_d09_mJB4_ch06',
        'gJB_sBM_cAll_d09_mJB4_ch07',
        'gJB_sBM_cAll_d09_mJB4_ch09',
        'gJB_sBM_cAll_d09_mJB5_ch09',
        'gJS_sFM_cAll_d01_mJS0_ch01',
        'gJS_sFM_cAll_d01_mJS1_ch02',
        'gJS_sFM_cAll_d02_mJS0_ch08',
        'gJS_sFM_cAll_d03_mJS0_ch01',
        'gJS_sBM_cAll_d03_mJS3_ch10',
        'gKR_sBM_cAll_d30_mKR5_ch02',
        'gKR_sBM_cAll_d30_mKR5_ch01',
        'gHO_sFM_cAll_d20_mHO5_ch13',
        'gWA_sBM_cAll_d27_mWA4_ch02',
        'gWA_sBM_cAll_d27_mWA4_ch08',
        'gWA_sFM_cAll_d26_mWA2_ch10',
        'gWA_sBM_cAll_d25_mWA3_ch04',
        'gWA_sFM_cAll_d27_mWA1_ch16',
        'gWA_sBM_cAll_d25_mWA1_ch04',
        'gWA_sBM_cAll_d27_mWA5_ch01',
        'gWA_sBM_cAll_d27_mWA5_ch08',
        'gWA_sBM_cAll_d27_mWA3_ch01',
        'gWA_sBM_cAll_d27_mWA2_ch08',
        'gWA_sBM_cAll_d26_mWA1_ch01',
        'gWA_sBM_cAll_d26_mWA0_ch09',
        'gWA_sBM_cAll_d27_mWA2_ch01',
        'gWA_sBM_cAll_d25_mWA2_ch04',
        'gWA_sBM_cAll_d25_mWA2_ch03',
    ]

    slc_videos = get_video_basename()
    for video in slc_videos:
        if video in ignore_lst:
            print(video)


def get_video_basename():
    video_basename = [
    'gJS_sFM_cAll_d01_mJS5_ch06',
    'gJS_sFM_cAll_d02_mJS3_ch04',
    'gJS_sFM_cAll_d03_mJS5_ch13',
    'gBR_sFM_cAll_d04_mBR0_ch01',
    'gBR_sFM_cAll_d05_mBR1_ch08',
    'gBR_sFM_cAll_d06_mBR2_ch16',
    'gJB_sFM_cAll_d07_mJB2_ch03',
    'gJB_sFM_cAll_d08_mJB2_ch10',
    'gJB_sFM_cAll_d09_mJB1_ch21',
    'gPO_sFM_cAll_d10_mPO3_ch07',
    'gPO_sFM_cAll_d11_mPO2_ch10',
    'gPO_sFM_cAll_d12_mPO1_ch16',
    'gLO_sFM_cAll_d13_mLO0_ch01',
    'gLO_sFM_cAll_d14_mLO2_ch10',
    'gLO_sFM_cAll_d15_mLO0_ch15',
    'gLH_sFM_cAll_d16_mLH2_ch03',
    'gLH_sFM_cAll_d17_mLH1_ch09',
    'gLH_sFM_cAll_d18_mLH1_ch16',
    'gHO_sFM_cAll_d19_mHO4_ch05',
    'gHO_sFM_cAll_d20_mHO3_ch11',
    'gHO_sFM_cAll_d21_mHO3_ch18',
    'gMH_sFM_cAll_d22_mMH2_ch03',
    'gMH_sFM_cAll_d23_mMH4_ch12',
    'gMH_sFM_cAll_d24_mMH1_ch16',
    'gWA_sFM_cAll_d25_mWA2_ch03',
    'gWA_sFM_cAll_d26_mWA1_ch09',
    'gWA_sFM_cAll_d27_mWA4_ch19',
    'gKR_sFM_cAll_d28_mKR4_ch05',
    'gKR_sFM_cAll_d29_mKR3_ch11',
    'gKR_sFM_cAll_d30_mKR5_ch20',
    ]
    return video_basename


def get_img_paths(base_img_pth):
    all_ims = []
    ims = glob.glob(os.path.join(base_img_pth, '*.jpg'))
    ims = np.array(sorted(ims))
    all_ims.append(ims)
    all_ims = np.stack(all_ims, axis=1)
    return all_ims


def batch_project(xyz, K, RT):
    """
    xyz: [B, N, 3]
    K: [B, 3, 3]
    RT: [B, 3, 4]
    """
    xyz = torch.matmul(xyz, RT[:,:,:3].transpose(1,2)) + RT[:,:,3:].transpose(1,2)
    xyz = torch.matmul(xyz, K.transpose(1,2))
    xy = xyz[:,:,:2] / xyz[:,:,2:]
    return xy


def test_rproj(id):
    import imageio
    rproj = './AIST_slc/vis/'
    img_pth = './AIST_slc/imgs/d{:02d}/c01/000000.jpg'.format(id)
    new_vertices_pth = './AIST_slc/imgs/d{:02d}/new_vertices/0.npy'.format(id)
    new_params_pth = './AIST_slc/imgs/d{:02d}/new_params/0.npy'.format(id)
    annots_pth = './AIST_slc/imgs/d{:02d}/annots.npy'.format(id)
    img = imageio.imread(img_pth).astype(np.float32) / 255.  
    new_vertices = np.load(new_vertices_pth) 
    new_params = np.load(new_params_pth, allow_pickle=True) 
    annots = np.load(annots_pth, allow_pickle=True)

    K = annots.item()['cams']['K'][0][None, ...]
    R = annots.item()['cams']['R'][0][None, ...]
    T = annots.item()['cams']['T'][0][None, ...]/1000
    RT = np.concatenate((R, T), axis=2)
    xyz = new_vertices[None, ...]

    proj_xy = batch_project(torch.tensor(xyz).float(), torch.tensor(K).float(), torch.tensor(RT).float())
    proj_xy = proj_xy[0].numpy()[::100]

    # visualize projected points for feature extraction
    import matplotlib.pyplot as plt
    j_s = proj_xy[:,0]
    i_s = proj_xy[:,1]
    plt.imshow(img)
    plt.plot(j_s, i_s, 'ro')
    plt.show()

    # back to tpose
    # print(np.min(new_vertices, axis=0))
    # print(np.max(new_vertices, axis=0))

    # transform to T-pose? # not implemented
    poses = new_params['poses']
    Rh = new_params['Rh']
    Th = new_params['Th']

    return None


def get_maskrcnn_model(mode='pointrend'):
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog
    if mode=='maskrcnn':
        
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
    
    if mode=='pointrend':

        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog
        coco_metadata = MetadataCatalog.get("coco_2017_val")

        # import PointRend project
        from detectron2.projects import point_rend

        cfg = get_cfg()
        # Add PointRend-specific config
        point_rend.add_pointrend_config(cfg)
        # Load a config from file
        cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
        cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
        predictor = DefaultPredictor(cfg)

    return predictor


from PIL import Image

def run(max_frames=600, num_cams=9):
    video_basenames = get_video_basename()
    predictor = get_maskrcnn_model()
    for video_basename in video_basenames[0:1]:
        annots = {'ims': []}
        for i in range(1, num_cams+1):
            video_name = video_basename[:9]+'0'+str(i)+video_basename[12:]
            src_video_pth = os.path.join( dst_video_pth, video_name+'.mp4')
            extract_video(src_video_pth, dst_img_pth, max_frames)

        performer_id = re.split('_', video_basename)[3]
        num_imgs = len(glob.glob(os.path.join('./AIST_slc/imgs', performer_id, 'c01', '*.jpg')))
        img_list = glob.glob(os.path.join('./AIST_slc/imgs', performer_id, 'c01', '*.jpg'))
        img_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        for i in range(num_imgs):
            frame_idx = int(os.path.basename(img_list[i])[:-4])
            dic = {}
            dic['ims'] = ['c0{}/{:06d}.jpg'.format(c, frame_idx) for c in range(1,10)]
            annots['ims'].append(dic)

        ################################## mask #####################################
        for cam_id in range(1, num_cams+1):
            if not os.path.exists( os.path.join(dst_img_pth, performer_id, 'mask', 'c0'+str(cam_id) ) ):
                os.makedirs(os.path.join(dst_img_pth, performer_id, 'mask', 'c0'+str(cam_id) ), exist_ok=True)
            img_list = glob.glob(os.path.join('./AIST_slc/imgs', performer_id, 'c0'+str(cam_id), '*.jpg'))

            for img_path in img_list:
                # run mask rcnn
                img = cv2.imread(img_path)
                outputs = predictor(img)

                mask = outputs["instances"].pred_masks
                c = outputs["instances"].pred_classes
                print(img_path, mask.shape, c)
                if 0 in c:
                    c_idx = torch.where(c==0)
                else:
                    c_idx = torch.where(c==c[0])
                mask = mask[c_idx]

                mask = mask.unsqueeze(-1).repeat(1,1,1,3).detach().cpu().numpy()[0]

                # save segmentation
                mask = Image.fromarray((mask * 255).astype(np.uint8))
                mask.save(os.path.join(dst_img_pth, performer_id, 'mask', \
                                       'c0'+str(cam_id), os.path.basename(img_path)[:-4]+'.png'))
        ################################## mask #####################################

        save_param_pth = os.path.join('./AIST_slc/imgs', performer_id, 'new_params' )
        save_vertices_pth = os.path.join('./AIST_slc/imgs', performer_id, 'new_vertices' )

        if not os.path.exists(save_param_pth):
            os.makedirs(save_param_pth, exist_ok=True)
        if not os.path.exists(save_vertices_pth):
            os.makedirs(save_vertices_pth, exist_ok=True)

        new_params, new_vertices = get_smpl_params(video_basename)
        img_list = glob.glob(os.path.join('./AIST_slc/imgs', performer_id, 'c01', '*.jpg'))
        img_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        for i in range(num_imgs):
            frame_idx = int(os.path.basename(img_list[i])[:-4])
            save_params = {}
            save_params['poses'] = new_params['poses'][frame_idx].reshape(1,72)
            save_params['Rh'] = new_params['Rh'][frame_idx].reshape(1,3)
            save_params['Th'] = new_params['Th'][frame_idx].reshape(1,3) / new_params['scaling']
            save_params['shapes'] = np.zeros((1,10))
            save_params['scaling'] = new_params['scaling']
            save_vertices = new_vertices[frame_idx] / new_params['scaling']

            np.save(os.path.join(save_param_pth, '{}'.format(frame_idx)), save_params)
            np.save(os.path.join(save_vertices_pth, '{}'.format(frame_idx)), save_vertices)

        cams = get_cams(video_basename, trans_scale = new_params['scaling'])
        annots['cams'] = cams
        save_annot_pth = os.path.join('./AIST_slc/imgs/', performer_id, 'annots.npy')
        np.save(save_annot_pth, annots)



if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--check_ignore', action='store_true')
    parser.add_argument('--test_rproj', action='store_true')

    args = parser.parse_args()

    if args.generate:
        run()
    if args.check_ignore:
        check_ignore_list()
    if args.test_rproj:
        test_rproj(id=1)

    def plot_smpl_3d_snippet():
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt
        data = np.load('d01/lbs/tvertices.npy', allow_pickle=True)
        ax = plt.axes(projection='3d')
        ax.scatter3D(data[:,0], data[:,1], data[:,2], cmap='Greens')
        plt.show()