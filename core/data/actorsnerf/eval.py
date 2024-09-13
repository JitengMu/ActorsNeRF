import os
import pickle
import re

import random
import numpy as np
import cv2
import torch
import torch.utils.data

from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

from configs import cfg


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_list):

        self.dataset_path = {}
        self.image_dir = {}
        self.canonical_joints = {}
        self.canonical_bbox = {}
        self.motion_weights_priors = {}
        self.frame_list = {}
        self.cameras = {}
        self.train_mesh_info = {}
        self.bgcolor = {}
        self.keyfilter = {}
        self.src_type = {}

        self.idx_hist = {}
        total_frames = 0
        for args in dataset_list:
            dataset_path = args['dataset_path']
            print('[Dataset Path]', dataset_path) 
            frame_start = args['frame_start']
            frame_end = args['frame_end']
            keyfilter= args['keyfilter']
            skip= args['skip']
            bgcolor= args['bgcolor']
            src_type= args['src_type']

            sub_idx = os.path.basename(dataset_path)

            if cfg.task == 'zju_mocap' or ('smpl_mocap' in cfg.task):
                # cam_idxs = [3,15]
                cam_idxs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
            elif cfg.task == 'AIST_mocap':
                #cam_idxs = [3,6]
                cam_idxs = [1,2,3,4,5,6,7]
            else:
                raise Exception("Not implemented")

            self.dataset_path[sub_idx] = {}
            self.image_dir[sub_idx] = {}
            self.canonical_joints[sub_idx] = {}
            self.canonical_bbox[sub_idx] = {}
            self.motion_weights_priors[sub_idx] = {}
            self.frame_list[sub_idx] = {}
            self.cameras[sub_idx] = {}
            self.train_mesh_info[sub_idx] = {}
            if cfg.task == 'zju_mocap' or ('smpl_mocap' in cfg.task):
                self.bgcolor[sub_idx] = bgcolor if bgcolor is not None else [0., 0., 0.] # sub
            elif cfg.task == 'AIST_mocap':
                self.bgcolor[sub_idx] = bgcolor if bgcolor is not None else [255., 255., 255.] # sub
            self.keyfilter[sub_idx] = keyfilter
            self.src_type[sub_idx] = src_type
    
            dataset_path_ = dataset_path
            first_camera = True
            for cam_idx in cam_idxs:
                dataset_path = dataset_path_
                dataset_path = os.path.join(dataset_path, str(cam_idx))
                image_dir = os.path.join(dataset_path, 'images')

                if first_camera:
                    canonical_joints, canonical_bbox = \
                        self.load_canonical_joints(dataset_path)
            
                    if 'motion_weights_priors' in keyfilter:
                        motion_weights_priors = \
                            approx_gaussian_bone_volumes(
                                canonical_joints, 
                                canonical_bbox['min_xyz'],
                                canonical_bbox['max_xyz'],
                                grid_size=cfg.mweight_volume.volume_size).astype('float32')
        
                cameras = self.load_train_cameras(dataset_path)
                mesh_infos = self.load_train_mesh_infos(dataset_path)
        
                framelist = self.load_train_frames(dataset_path) 
                framelist = framelist[frame_start: frame_end]
                select_framelist = framelist[::skip]
                num_frames = len(select_framelist)

                select_camera = {}
                select_train_mesh_info = {}
                for i, frame_idx in enumerate(select_framelist):
                    self.idx_hist[i + total_frames] = {'sub_idx': sub_idx, 'cam_idx': cam_idx, 'frame_idx': frame_idx}
                    select_camera[frame_idx] = cameras[frame_idx]
                    select_train_mesh_info[frame_idx] = mesh_infos[frame_idx]
                total_frames += num_frames
                print(sub_idx, cam_idx, total_frames)
        
                self.dataset_path[sub_idx][cam_idx] = dataset_path # sub, cam
                self.image_dir[sub_idx][cam_idx] = image_dir # sub, cam
                if first_camera:
                    self.canonical_joints[sub_idx] = canonical_joints # sub
                    self.canonical_bbox[sub_idx] = canonical_bbox # sub
                    if 'motion_weights_priors' in keyfilter:
                        self.motion_weights_priors[sub_idx] = motion_weights_priors # sub
                self.frame_list[sub_idx][cam_idx] = select_framelist # sub, cam, frame
                self.cameras[sub_idx][cam_idx] = select_camera # sub, cam, frame
                self.train_mesh_info[sub_idx][cam_idx] = select_train_mesh_info # sub, cam, frame
                first_camera = False

        self.total_frames = total_frames
        print(f' -- Total Rendered Frames: {self.total_frames}')


        # multiview dataset
        if cfg.use_data_cross_pose or cfg.use_data_cross_view:
            self.enc_dataset_path = {}
            self.enc_image_dir = {}
            self.enc_canonical_joints = {}
            self.enc_canonical_bbox = {}
            self.enc_motion_weights_priors = {}
            self.enc_frame_list = {}
            self.enc_cameras = {}
            self.enc_train_mesh_info = {}
            self.enc_bgcolor = {}
            self.enc_keyfilter = {}
            self.enc_src_type = {}

            self.enc_idx_hist = {}
            self.enc_frame_to_idx = {}
            total_frames = 0
            for args in dataset_list:
                dataset_path = args['dataset_path']
                print('[Dataset Path]', dataset_path) 
                # frame_start = args['frame_start']
                # frame_end = args['frame_end']
                keyfilter= args['keyfilter']
                # skip= args['skip']
                bgcolor= args['bgcolor']
                src_type= args['src_type']
                self.ray_shoot_mode = args['ray_shoot_mode']
                maxframes = -1
                if 'maxframes' in args:
                    maxframes = args['maxframes']

                ##### load all frames but only use a few depending on the setting #####
                frame_start = 0
                frame_end = -1
                #######################################################################

                sub_idx = os.path.basename(dataset_path)

                if cfg.task == 'zju_mocap' or ('smpl_mocap' in cfg.task):
                    cam_idxs = [0]
                elif cfg.task == 'AIST_mocap':
                    cam_idxs = [0]

                self.enc_dataset_path[sub_idx] = {}
                self.enc_image_dir[sub_idx] = {}
                self.enc_canonical_joints[sub_idx] = {}
                self.enc_canonical_bbox[sub_idx] = {}
                self.enc_motion_weights_priors[sub_idx] = {}
                self.enc_frame_list[sub_idx] = {}
                self.enc_cameras[sub_idx] = {}
                self.enc_train_mesh_info[sub_idx] = {}
                if cfg.task == 'zju_mocap' or ('smpl_mocap' in cfg.task):
                    self.enc_bgcolor[sub_idx] = bgcolor if bgcolor is not None else [0., 0., 0.] # sub
                elif cfg.task == 'AIST_mocap':
                    self.enc_bgcolor[sub_idx] = bgcolor if bgcolor is not None else [255., 255., 255.] # sub
                self.enc_keyfilter[sub_idx] = keyfilter
                self.enc_src_type[sub_idx] = src_type
        
                dataset_path_ = dataset_path
                first_camera = True
                for cam_idx in cam_idxs:
                    dataset_path = dataset_path_
                    dataset_path = os.path.join(dataset_path, str(cam_idx))
                    image_dir = os.path.join(dataset_path, 'images')

                    if first_camera:
                        canonical_joints, canonical_bbox = \
                            self.load_canonical_joints(dataset_path)
                
                        if 'motion_weights_priors' in keyfilter:
                            motion_weights_priors = \
                                approx_gaussian_bone_volumes(
                                    canonical_joints, 
                                    canonical_bbox['min_xyz'],
                                    canonical_bbox['max_xyz'],
                                    grid_size=cfg.mweight_volume.volume_size).astype('float32')
            
                    cameras = self.load_train_cameras(dataset_path)
                    mesh_infos = self.load_train_mesh_infos(dataset_path)
            
                    framelist = self.load_train_frames(dataset_path) 
                    framelist = framelist[frame_start: frame_end]
                    skip = 1
                    select_framelist = framelist[::skip]
                    num_frames = len(select_framelist)

                    select_camera = {}
                    select_train_mesh_info = {}
                    for i, frame_idx in enumerate(select_framelist):
                        self.enc_idx_hist[i + total_frames] = {'sub_idx': sub_idx, 'cam_idx': cam_idx, 'frame_idx': frame_idx}
                        self.enc_frame_to_idx[(sub_idx, cam_idx, frame_idx)] = i + total_frames
                        select_camera[frame_idx] = cameras[frame_idx]
                        select_train_mesh_info[frame_idx] = mesh_infos[frame_idx]
                    total_frames += num_frames
                    print(sub_idx, cam_idx, total_frames)
            
                    self.enc_dataset_path[sub_idx][cam_idx] = dataset_path # sub, cam
                    self.enc_image_dir[sub_idx][cam_idx] = image_dir # sub, cam
                    if first_camera:
                        self.enc_canonical_joints[sub_idx] = canonical_joints # sub
                        self.enc_canonical_bbox[sub_idx] = canonical_bbox # sub
                        if 'motion_weights_priors' in keyfilter:
                            self.enc_motion_weights_priors[sub_idx] = motion_weights_priors # sub
                    self.enc_frame_list[sub_idx][cam_idx] = select_framelist # sub, cam, frame
                    self.enc_cameras[sub_idx][cam_idx] = select_camera # sub, cam, frame
                    self.enc_train_mesh_info[sub_idx][cam_idx] = select_train_mesh_info # sub, cam, frame
                    first_camera = False



    def load_canonical_joints(self, dataset_path):
        cl_joint_path = os.path.join(dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_canonical_bbox(canonical_joints)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self, dataset_path):
        cameras = None
        with open(os.path.join(dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    @staticmethod
    def skeleton_to_canonical_bbox(skeleton):
        bbox_offset = np.max(skeleton, axis=0) - np.min(skeleton, axis=0)
        bbox_offset[0] = bbox_offset[0]*0.2
        bbox_offset[1] = bbox_offset[1]*0.2
        min_xyz = np.min(skeleton, axis=0) - bbox_offset
        max_xyz = np.max(skeleton, axis=0) + bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self, dataset_path):
        mesh_infos = None
        with open(os.path.join(dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos

    def load_train_frames(self, dataset_path):
        img_paths = list_files(os.path.join(dataset_path, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]

    def query_dst_skeleton(self, sub_idx, cam_idx, frame_idx):
        return {
            'poses': self.train_mesh_info[sub_idx][cam_idx][frame_idx]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.train_mesh_info[sub_idx][cam_idx][frame_idx]['tpose_joints'].astype('float32'),
            'bbox': self.train_mesh_info[sub_idx][cam_idx][frame_idx]['bbox'].copy(),
            'Rh': self.train_mesh_info[sub_idx][cam_idx][frame_idx]['Rh'].astype('float32'),
            'Th': self.train_mesh_info[sub_idx][cam_idx][frame_idx]['Th'].astype('float32')
        }

    def query_dst_skeleton_enc(self, sub_idx, cam_idx, frame_idx):
        return {
            'poses': self.enc_train_mesh_info[sub_idx][cam_idx][frame_idx]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.enc_train_mesh_info[sub_idx][cam_idx][frame_idx]['tpose_joints'].astype('float32'),
            'bbox': self.enc_train_mesh_info[sub_idx][cam_idx][frame_idx]['bbox'].copy(),
            'Rh': self.enc_train_mesh_info[sub_idx][cam_idx][frame_idx]['Rh'].astype('float32'),
            'Th': self.enc_train_mesh_info[sub_idx][cam_idx][frame_idx]['Th'].astype('float32')
        }



#    def get_freeview_camera(self, frame_idx, total_frames, trans=None):
#        E = rotate_camera_by_frame_idx(
#                extrinsics=self.train_camera['extrinsics'], 
#                frame_idx=frame_idx,
#                period=total_frames,
#                trans=trans,
#                **self.ROT_CAM_PARAMS[self.src_type])
#        K = self.train_camera['intrinsics'].copy()
#        K[:2] *= cfg.resize_img_scale
#        return K, E

    def load_image(self, sub_idx, cam_idx, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir[sub_idx][cam_idx], '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path[sub_idx][cam_idx], 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))
        
        if 'distortions' in self.cameras[sub_idx][cam_idx][frame_name]:
            K = self.cameras[sub_idx][cam_idx][frame_name]['intrinsics']
            D = self.cameras[sub_idx][cam_idx][frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                             fx=cfg.resize_img_scale,
                             fy=cfg.resize_img_scale,
                             interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask

    def load_image_enc(self, sub_idx, cam_idx, frame_name, bg_color):
        imagepath = os.path.join(self.enc_image_dir[sub_idx][cam_idx], '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.enc_dataset_path[sub_idx][cam_idx], 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))
        
#        if 'distortions' in self.enc_cameras[sub_idx][cam_idx][frame_name]:
#            K = self.enc_cameras[sub_idx][cam_idx][frame_name]['intrinsics']
#            D = self.enc_cameras[sub_idx][cam_idx][frame_name]['distortions']
#            orig_img = cv2.undistort(orig_img, K, D)
#            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                             fx=cfg.resize_img_scale,
                             fy=cfg.resize_img_scale,
                             interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask


    def load_smpl_vertices(self, sub_idx, cam_idx, frame_name, dst_skel_info):
        i = int(frame_name[-6:])
        # read xyz in the world coordinate system
        vertices_path = os.path.join(os.path.dirname(self.enc_dataset_path[sub_idx][cam_idx]), 'new_vertices',
                                     '{}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        Rh = dst_skel_info['Rh'].astype(np.float32)
        Th = dst_skel_info['Th'].astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)
        if 'smpl_mocap' in cfg.task:
            pxyz = wxyz
        return pxyz

    def prepare_input(self, sub_idx, cam_idx, frame_name, dst_skel_info):
        i = int(frame_name[-6:])
        # read xyz in the world coordinate system
        vertices_path = os.path.join(os.path.dirname(self.dataset_path[sub_idx][cam_idx]), 'new_vertices',
                                     '{}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        Rh = dst_skel_info['Rh'].astype(np.float32)
        Th = dst_skel_info['Th'].astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)
        if 'smpl_mocap' in cfg.task:
            pxyz = wxyz

        # pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
        # pbw = pbw.astype(np.float32)
        tvertices_path = os.path.join(os.path.dirname(self.dataset_path[sub_idx][cam_idx]), 'lbs/tvertices.npy')
        txyz = np.load(tvertices_path).astype(np.float32)

        # tbw_path = os.path.join(os.path.dirname(self.dataset_path[sub_idx][cam_idx]), 'lbs/tbw.npy')
        # tbw = np.load(tbw_path).astype(np.float32)

        tbbox = self.canonical_bbox[sub_idx]
        # joints = self.canonical_joints[sub_idx]

        # construct the coordinate
        dhw = txyz[:, [2, 1, 0]]
        min_xyz = tbbox['min_xyz']
        max_xyz = tbbox['max_xyz']
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        txyz_coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        txyz_shape = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        txyz_shape = (txyz_shape | (8 - 1)) + 1
        txyz_dic = {'txyz': txyz, 
                    'txyz_coord': txyz_coord,
                    'txyz_shape': txyz_shape,
                    }

        # construct the pxyz coordinate
        dhw = pxyz[:, [2, 1, 0]]
        min_xyz = np.min(pxyz, axis=0) - 0.05
        max_xyz = np.max(pxyz, axis=0) + 0.05
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        pxyz_coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        pxyz_shape = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        pxyz_shape = (pxyz_shape | (32 - 1)) + 1

        pxyz_dic = {'pxyz': pxyz, 
                    'pxyz_coord': pxyz_coord,
                    'pxyz_shape': pxyz_shape,
                    'pxyz_bbox_min': np.min(pxyz, axis=0) - 0.05,
                    'pxyz_bbox_max': np.max(pxyz, axis=0) + 0.05,
                    }

        return pxyz, txyz_dic, pxyz_dic

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        sub_idx = self.idx_hist[idx]['sub_idx']
        cam_idx = self.idx_hist[idx]['cam_idx']
        frame_idx = self.idx_hist[idx]['frame_idx']
        frame_name = frame_idx
        results = {
            'sub_idx': sub_idx,
            'cam_idx': cam_idx,
            'frame_name': frame_name,
        }

        bgcolor = np.array(self.bgcolor[sub_idx], dtype='float32')

        img, alpha = self.load_image(sub_idx, cam_idx, frame_idx, bgcolor)
        img = (img / 255.).astype('float32')
        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(sub_idx, cam_idx, frame_idx)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_Rh = dst_skel_info['Rh']
        dst_Th = dst_skel_info['Th']

        smpl_vertices, txyz_dic, pxyz_dic = self.prepare_input(sub_idx, cam_idx, frame_name, dst_skel_info)
        #dst_bbox['min_xyz'] = pxyz_dic['pxyz_bbox_min']
        #dst_bbox['max_xyz'] = pxyz_dic['pxyz_bbox_max']

        assert frame_idx in self.cameras[sub_idx][cam_idx]
        K = self.cameras[sub_idx][cam_idx][frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= cfg.resize_img_scale

        E = self.cameras[sub_idx][cam_idx][frame_idx]['extrinsics']

#        K, E = self.get_freeview_camera(
#                        frame_idx=idx,
#                        total_frames=self.total_frames,
#                        trans=dst_Th)
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_Rh,
                Th=dst_Th)
        R = E[:3, :3]
        T = E[:3, 3]
        cam_R = E[:3, :3]
        cam_T = E[:3, 3].reshape(3, 1)

        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter[sub_idx]:
            results.update({
                'img': img,
                'smpl_vertices': smpl_vertices,
                'dst_bbox': dst_bbox,
                'txyz_dic': txyz_dic,
                'pxyz_dic': pxyz_dic,
                'cam_K': K,
                'cam_R': cam_R,
                'cam_T': cam_T,
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})

        if 'target_rgbs' in self.keyfilter[sub_idx]:
            results['target_rgbs'] = img

        if 'motion_bases' in self.keyfilter[sub_idx]:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints)
            cnl_gtfms = get_canonical_global_tfms(self.canonical_joints[sub_idx])
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })

        if 'motion_weights_priors' in self.keyfilter[sub_idx]:
            results['motion_weights_priors'] = self.motion_weights_priors[sub_idx].copy()

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter[sub_idx]:
            min_xyz = self.canonical_bbox[sub_idx]['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox[sub_idx]['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter[sub_idx]:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })

        # multiview data
        data_enc = {}
        num_aug_frames = 3
        skip = int(re.split('skip', cfg.logdir)[-1])
        if results['sub_idx']=='387':
            if skip==1:
                 render_frame_names = ['frame_000075', 'frame_000225', 'frame_000300']
            if skip==3:
                 render_frame_names = ['frame_000075', 'frame_000225', 'frame_000300']
            if skip==5:
                 render_frame_names = ['frame_000075', 'frame_000225', 'frame_000300']
            if skip==10:
                 render_frame_names = ['frame_000080', 'frame_000230', 'frame_000300']
            if skip==30:
                 render_frame_names = ['frame_000090', 'frame_000240', 'frame_000300']
            if skip==60:
                render_frame_names = ['frame_000075', 'frame_000225', 'frame_000300']

        if results['sub_idx']=='393':
            if skip==1:
                 render_frame_names = ['frame_000075', 'frame_000150', 'frame_000300']
            if skip==3:
                 render_frame_names = ['frame_000075', 'frame_000150', 'frame_000300']
            if skip==5:
                 render_frame_names = ['frame_000075', 'frame_000150', 'frame_000300']
            if skip==10:
                 render_frame_names = ['frame_000080', 'frame_000150', 'frame_000300']
            if skip==30:
                 render_frame_names = ['frame_000090', 'frame_000150', 'frame_000300']
            if skip==60:
                render_frame_names = ['frame_000075', 'frame_000150', 'frame_000300']

        if results['sub_idx']=='394':
            if skip==1:
                 render_frame_names = ['frame_000001', 'frame_000150', 'frame_000300']
            if skip==3:
                 render_frame_names = ['frame_000003', 'frame_000150', 'frame_000300']
            if skip==5:
                 render_frame_names = ['frame_000005', 'frame_000150', 'frame_000300']
            if skip==10:
                 render_frame_names = ['frame_000010', 'frame_000150', 'frame_000300']
            if skip==30:
                 render_frame_names = ['frame_000030', 'frame_000150', 'frame_000300']
            if skip==60:
                render_frame_names = ['frame_000000', 'frame_000150', 'frame_000300']

        ###### AIST #####
        elif cfg.task == 'AIST_mocap':
            if results['sub_idx']=='d16':
                render_frame_names = ['frame_000200', 'frame_000832', 'frame_000840']
            if results['sub_idx']=='d17':
                render_frame_names = ['frame_000564', 'frame_001228', 'frame_001252']
            if results['sub_idx']=='d18':
                render_frame_names = ['frame_000200', 'frame_000708', 'frame_000736']
            if results['sub_idx']=='d19':
                render_frame_names = ['frame_000212', 'frame_000408', 'frame_000556']
            if results['sub_idx']=='d20':
                render_frame_names = ['frame_000200', 'frame_000700', 'frame_000712']
        if cfg.use_data_cross_pose or cfg.use_data_cross_view:
            for aug_idx in range(num_aug_frames):

                render_sub_idx = results['sub_idx']
                # render_cam_idx = results['cam_idx']
                render_cam_idx = 0
                # render_frame_name = np.random.choice(self.enc_frame_list[render_sub_idx][render_cam_idx], 1).item()
                render_frame_name = render_frame_names[aug_idx]
                idx = self.enc_frame_to_idx[(render_sub_idx, render_cam_idx, render_frame_name)]
                # print(render_sub_idx, cam_idxs[aug_idx], render_frame_name, idx, frame_idx)

                sub_idx = self.enc_idx_hist[idx]['sub_idx']
                cam_idx = self.enc_idx_hist[idx]['cam_idx']
                frame_idx = self.enc_idx_hist[idx]['frame_idx']
                frame_name = frame_idx
                data_enc[aug_idx] = {}
                data_enc[aug_idx]['sub_idx'] = sub_idx
                data_enc[aug_idx]['cam_idx'] = cam_idx
                data_enc[aug_idx]['frame_name'] = frame_name

                bgcolor = np.array(self.enc_bgcolor[sub_idx], dtype='float32')

                img, mask = self.load_image_enc(sub_idx, cam_idx, frame_idx, bgcolor)
                img = (img / 255.).astype('float32')
                H, W = img.shape[0:2]

                dst_skel_info = self.query_dst_skeleton_enc(sub_idx, cam_idx, frame_idx)
                dst_bbox = dst_skel_info['bbox']
                dst_poses = dst_skel_info['poses']
                dst_tpose_joints = dst_skel_info['dst_tpose_joints']
                dst_Rh = dst_skel_info['Rh']
                dst_Th = dst_skel_info['Th']

                smpl_vertices = self.load_smpl_vertices(render_sub_idx, render_cam_idx, frame_name, dst_skel_info)

                assert frame_idx in self.enc_cameras[sub_idx][cam_idx]
                K = self.enc_cameras[sub_idx][cam_idx][frame_name]['intrinsics'][:3, :3].copy()
                K[:2] *= cfg.resize_img_scale

                E = self.enc_cameras[sub_idx][cam_idx][frame_idx]['extrinsics']
                E = apply_global_tfm_to_camera(
                        E=E, 
                        Rh=dst_Rh,
                        Th=dst_Th)
                R = E[:3, :3]
                T = E[:3, 3]
                cam_R = E[:3, :3]
                cam_T = E[:3, 3].reshape(3, 1)

                if 'rays' in self.enc_keyfilter[sub_idx]:
                    data_enc[aug_idx]['img'] = img
                    data_enc[aug_idx]['mask'] = mask
                    data_enc[aug_idx]['cam_K'] = K.copy()
                    data_enc[aug_idx]['cam_R'] = cam_R
                    data_enc[aug_idx]['cam_T'] = cam_T
                    data_enc[aug_idx]['bgcolor'] = bgcolor
                    data_enc[aug_idx]['smpl_vertices'] = smpl_vertices

                if 'motion_bases' in self.enc_keyfilter[sub_idx]:
                    dst_Rs, dst_Ts = body_pose_to_body_RTs(
                            dst_poses, dst_tpose_joints)
                    cnl_gtfms = get_canonical_global_tfms(self.canonical_joints[sub_idx])
                    data_enc[aug_idx]['dst_Rs'] = dst_Rs
                    data_enc[aug_idx]['dst_Ts'] = dst_Ts
                    data_enc[aug_idx]['cnl_gtfms'] = cnl_gtfms

            results["data_enc"] = data_enc

        return results, idx
