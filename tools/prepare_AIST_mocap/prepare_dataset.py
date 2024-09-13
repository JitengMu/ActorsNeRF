import os
import sys

from shutil import copyfile
import shutil

import pickle
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

#from third_parties.smpl.smpl_numpy import SMPL
from smplx import SMPL # note that this is the modified smplx repo
# from core.utils.file_util import split_path
# from core.utils.image_util import load_image, save_image, to_3ch_image

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'd01.yaml',
                    'the path of config file')

# MODEL_DIR = '../../third_parties/smpl/models'

MODEL_DIR = 'imgs/'
with open(MODEL_DIR + 'SMPL_NEUTRAL.pkl', 'rb') as smpl_file:
    smpl_data = pickle.load(smpl_file,
                                       encoding='latin1')

J_regressor = smpl_data["J_regressor_prior"]

def world_points_to_pose_points(wpts, Rh, Th):
    """
    wpts: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = np.matmul(wpts - Th, Rh)
    return pts

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


def load_image(path, to_rgb=True):
    img = Image.open(path)
    return img.convert('RGB') if to_rgb else img


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

    
def to_8b_image(image):
    return (255.* np.clip(image, 0., 1.)).astype(np.uint8)


def to_3ch_image(image):
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        assert image.shape[2] == 1
        return np.concatenate([image, image, image], axis=-1)
    else:
        print(f"to_3ch_image: Unsupported Shapes: {len(image.shape)}")
        return image

    
def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    flag1 = False
    flag2 = False
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    if os.path.exists(msk_path):
        msk = np.array(load_image(msk_path))[:, :, 0]
        msk = (msk != 0).astype(np.uint8)
        flag1 = True

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    if os.path.exists(msk_path):
        msk_cihp = np.array(load_image(msk_path))[:, :, 0]
        msk_cihp = (msk_cihp != 0).astype(np.uint8)
        flag2 = True

    if flag1 and flag2:
        msk = (msk | msk_cihp).astype(np.uint8)
    elif flag2:
        msk = msk_cihp.astype(np.uint8)
    elif flag1:
        msk = msk.astype(np.uint8)
    msk[msk == 1] = 255

    return msk


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    dataset_dir = cfg['dataset']['AIST_mocap_path']
    subject_dir = os.path.join(dataset_dir, f"{subject}")
    src_dir = os.path.join(subject_dir, "new_vertices")
    trg_dir = os.path.join(cfg['output']['dir'], 
                                   subject if 'name' not in cfg['output'].keys() else cfg['output']['name'], 'new_vertices')
    shutil.copytree(src_dir, trg_dir)

    lbs_dir = os.path.join(cfg['output']['dir'], 
                                   subject if 'name' not in cfg['output'].keys() else cfg['output']['name'], 'lbs')
    shutil.copytree(os.path.join(subject_dir, 'lbs/'), lbs_dir)

    for select_view in range(9):
    #for select_view in range(21): # 313, 315
        cfg = parse_config()
        subject = cfg['dataset']['subject']
        sex = cfg['dataset']['sex']
        max_frames = cfg['max_frames']
    
        dataset_dir = cfg['dataset']['AIST_mocap_path']
        subject_dir = os.path.join(dataset_dir, f"{subject}")
        smpl_params_dir = os.path.join(subject_dir, "new_params")
    
        #select_view = cfg['training_view']
    
        anno_path = os.path.join(subject_dir, 'annots.npy')
        annots = np.load(anno_path, allow_pickle=True).item()
        
        # load cameras
        cams = annots['cams']
        cam_Ks = np.array(cams['K'])[select_view].astype('float32')
        cam_Rs = np.array(cams['R'])[select_view].astype('float32')
        cam_Ts = np.array(cams['T'])[select_view].astype('float32') / 1000.
#         cam_Ds = np.array(cams['D'])[select_view].astype('float32')
    
        K = cam_Ks     #(3, 3)
#         D = cam_Ds[:, 0]
        E = np.eye(4)  #(4, 4)
        cam_T = cam_Ts[:3, 0]
        E[:3, :3] = cam_Rs
        E[:3, 3]= cam_T
        
        # load image paths
        img_path_frames_views = annots['ims']
        img_paths = np.array([
            np.array(multi_view_paths['ims'])[select_view] \
                for multi_view_paths in img_path_frames_views
        ])
        if max_frames > 0:
            img_paths = img_paths[:max_frames]
    
        output_path = os.path.join(cfg['output']['dir'], 
                                   subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
        output_path = os.path.join(output_path, str(select_view))
        os.makedirs(output_path, exist_ok=True)
        out_img_dir  = prepare_dir(output_path, 'images')
        out_mask_dir = prepare_dir(output_path, 'masks')
    
        # copy config file
        copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))
    
        #smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)
        smpl_model = SMPL(model_path=MODEL_DIR, sex=sex, batch_size=1)
 
        cameras = {}
        mesh_infos = {}
        all_tpose_joints = []
        all_betas = []
        #for idx, ipath in enumerate(tqdm(img_paths[810:1350])): # 396
        for idx, ipath in enumerate(tqdm(img_paths)):
            out_name = 'frame_{:06d}'.format(int(os.path.basename(ipath)[:-4]))
    
            img_path = os.path.join(subject_dir, ipath)
        
            # load image
            img = np.array(load_image(img_path))
    
#            if subject in ['313', '315']:
#                _, image_basename, _ = split_path(img_path)
#                start = image_basename.find(')_')
#                smpl_idx = int(image_basename[start+2: start+6])
#                out_name = 'frame_{:06d}'.format(smpl_idx)
#            elif subject in ['396']:
#                smpl_idx = idx + 810
#                out_name = 'frame_{:06d}'.format(smpl_idx)
#            else:
            smpl_idx = int(os.path.basename(ipath)[:-4])
    
            # load smpl parameters
            smpl_params = np.load(
                os.path.join(smpl_params_dir, f"{smpl_idx}.npy"),
                allow_pickle=True).item()
    
            betas = smpl_params['shapes'][0] #(10,)
            poses = smpl_params['poses'][0]  #(72,)
            Rh = smpl_params['Rh'][0]  #(3,)
            Th = smpl_params['Th'][0]  #(3,)
            scaling = smpl_params['scaling']
            
            all_betas.append(betas)
    
            # write camera info
            cameras[out_name] = {
                    'intrinsics': K,
                    'extrinsics': E,
#                     'distortions': D
            }
    
            # write mesh info
            smpl_poses = poses.copy()[None, ...]
            smpl_poses[:,:3] = Rh.copy()[None, ...]
            smpl_trans = Th[None, ...] * scaling
            smpl_scaling = scaling.copy()
            vertices = smpl_model.forward(
                global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
                body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
                transl=torch.from_numpy(smpl_trans).float(),
                scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
                ).vertices.detach().numpy() / smpl_scaling

            smpl_poses = poses.copy()[None, ...]
            smpl_poses[:, :3] = Rh.copy()[None, ...]
            smpl_poses[:, 3:] = 0
            smpl_trans = Th[None, ...] * scaling
            smpl_scaling = scaling.copy()
            tpose_vertices = smpl_model.forward(
                global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
                body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
                transl=torch.from_numpy(smpl_trans).float(),
                scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
                ).vertices.detach().numpy() / smpl_scaling

            R = batch_rodrigues(Rh[None, ...])
            vertices = world_points_to_pose_points(vertices, R, Th.reshape(1,1,3))[0]
            tpose_vertices = world_points_to_pose_points(tpose_vertices, R, Th.reshape(1,1,3))[0]
            joints = J_regressor@vertices
            tpose_joints = J_regressor@tpose_vertices
#            print(np.min(joints, axis=0), np.max(joints, axis=0))
#            print(np.min(tpose_joints, axis=0), np.max(tpose_joints, axis=0))
            
#             _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
#             _, joints = smpl_model(poses, betas)
            all_tpose_joints.append(tpose_joints)
            
            
            mesh_infos[out_name] = {
                'Rh': Rh,
                'Th': Th,
                'poses': poses,
                'joints': joints, 
                'tpose_joints': tpose_joints
            }
    
            # load and write mask
            mask = get_mask(subject_dir, ipath)
            save_image(to_3ch_image(mask), 
                       os.path.join(out_mask_dir, out_name+'.png'))
    
            # write image
            out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
            save_image(img, out_image_path)
    
        # write camera infos
        with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
            pickle.dump(cameras, f)
            
        # write mesh infos
        with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
            pickle.dump(mesh_infos, f)
    
        # write canonical joints
#        avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
#        smpl_model = SMPL(sex, model_dir=MODEL_DIR)
#         _, template_joints = smpl_model(np.zeros(72), avg_betas)
#        template_vertices, _ = smpl_model(np.zeros(72), avg_betas)
        template_joints = np.mean(np.stack(all_tpose_joints, axis=0), axis=0)
        
        
        with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
            pickle.dump(
                {
                    'joints': template_joints,
                }, f)

if __name__ == '__main__':
    app.run(main)