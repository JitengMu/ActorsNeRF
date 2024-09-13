import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from core.utils.network_util import MotionBasisComputer
from core.nets.actorsnerf.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp, \
    load_encoder, \
    load_sparse_conv_net

import matplotlib.pyplot as plt

from configs import cfg

from spconv.pytorch.conv import (SparseConv2d, SparseConv3d,
                                SparseConvTranspose2d,
                                SparseConvTranspose3d, SparseInverseConv2d,
                                SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.identity import Identity
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.ops import ConvAlgo
from spconv.pytorch.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.pytorch.tables import AddTable, ConcatTable

with_batch_dim = False
def tpose_points_to_pose_points(pts, bw, A):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = pts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    pts = pts + A[..., :3, 3]
    return pts

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

def prepare_txyz_sp_input(txyz_dic, xyz):
    sp_input = {}

    # coordinate: [N, 4] --- batch_idx, z, y, x
    sh = txyz_dic['txyz_coord'].shape
    # idx = [torch.full([sh[1]], i) for i in range(sh[0])]
    # idx = torch.cat(idx).to(txyz_dic['txyz_coord'])
    idx = [torch.full([sh[1]], xyz.device.index)]
    idx = torch.cat(idx).to(xyz.device)
    coord = txyz_dic['txyz_coord'].view(-1, sh[-1]).to(xyz.device)
    sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1).int()

    out_sh, _ = torch.max(txyz_dic['txyz_shape'], dim=0)
    sp_input['out_sh'] = out_sh.tolist()
    sp_input['batch_size'] = sh[0]

    return sp_input

def prepare_pxyz_sp_input(pxyz_dic):
    sp_input = {}

    # coordinate: [N, 4], batch_idx, z, y, x
    sh = pxyz_dic['pxyz_coord'].shape
    idx = [torch.full([sh[1]], i) for i in range(sh[0])]
    idx = torch.cat(idx).to(pxyz_dic['pxyz_coord'])
    coord = pxyz_dic['pxyz_coord'].view(-1, sh[-1])
    sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

    out_sh, _ = torch.max(pxyz_dic['pxyz_shape'], dim=0)
    sp_input['out_sh'] = out_sh.tolist()
    sp_input['batch_size'] = sh[0]

    return sp_input

def get_grid_coords(pts, sp_input, cnl_bbox_min_xyz):
    # convert xyz to the voxel coordinate dhw
    dhw = pts[..., [2, 1, 0]]
    min_dhw = cnl_bbox_min_xyz[..., [2, 1, 0]]
    dhw = dhw - min_dhw
    dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
    # convert the voxel coordinate to [-1, 1]
    out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
    dhw = dhw / out_sh * 2 - 1
    # convert dhw to whd, since the occupancy is indexed by dhw
    grid_coords = dhw[..., [2, 1, 0]]
    return grid_coords[None, None, None]

def get_sparse_conv_input(xyz, txyz_dic, pxyz_dic, cnl_bbox_min_xyz, cnl_bbox_scale_xyz):
    ########### txyz ##################
    sp_input = prepare_txyz_sp_input(txyz_dic, xyz)

    if cfg.use_cnl_normalize_coords:
        _xyz =  (xyz + 1) / cnl_bbox_scale_xyz[None] + cnl_bbox_min_xyz[None]
        grid_coords = get_grid_coords(_xyz, sp_input, cnl_bbox_min_xyz[None])
    else:
        grid_coords = get_grid_coords(_xyz, sp_input, cnl_bbox_min_xyz[None])

    coord = sp_input['coord']
    out_sh = sp_input['out_sh']
    batch_size = sp_input['batch_size']

    return coord, out_sh, batch_size, grid_coords


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = \
            load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
                                        cfg.non_rigid_motion_mlp.i_embed)

        if cfg.use_non_rigid_motions:
            self.non_rigid_mlp = \
                load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
                    pos_embed_size=non_rigid_pos_embed_size,
                    condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
                    mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
                    mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
                    skips=cfg.non_rigid_motion_mlp.skips)
            self.non_rigid_mlp = \
                nn.DataParallel(
                    self.non_rigid_mlp,
                    device_ids=cfg.secondary_gpus,
                    output_device=cfg.primary_gpus[0])

        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires, 
                         cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn

        # canonical mlp 
        skips = [4]
        self.cnl_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=cnl_pos_embed_size, 
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=skips)
        self.cnl_mlp = \
            nn.DataParallel(
                self.cnl_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.primary_gpus[0]
                )

        # pose decoder MLP
        if cfg.use_pose_correction:
            self.pose_decoder = \
                load_pose_decoder(cfg.pose_decoder.module)(
                    embedding_size=cfg.pose_decoder.embedding_size,
                    mlp_width=cfg.pose_decoder.mlp_width,
                    mlp_depth=cfg.pose_decoder.mlp_depth)
    
        # encoder
        if cfg.use_enc:
            self.encoder = load_encoder(cfg.encoder.module)(
                    encoder_arch=cfg.encoder.encoder_arch)

        # smpl feat
        if cfg.use_smpl_feat:
            self.sparse_conv_net = load_sparse_conv_net(cfg.sparse_conv_net.module)()

    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])
        if cfg.use_non_rigid_motions:
            self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

        return self


    def _query_mlp(
            self,
            iter_val,
            pos_xyz,
            pts_mask_hard,
            cnl_bbox_min_xyz, 
            cnl_bbox_scale_xyz,
            pos_embed_fn, 
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input,
            enc_local_feat_lst=None,
            enc_local_feat_smpl_lst=None,
            proj_xy_lst=None,
            proj_xy_smpl_lst=None,
            txyz_dic=None,
            pxyz_dic=None,
            img=None,
            motion_weights_vol=None,
            ):
        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        iter_val,
                        pos_flat=pos_flat,
                        pts_mask_hard=pts_mask_hard,
                        pos_embed_fn=pos_embed_fn,
                        cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                        cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        enc_local_feat_lst=enc_local_feat_lst,
                        enc_local_feat_smpl_lst=enc_local_feat_smpl_lst,
                        proj_xy_lst=proj_xy_lst,
                        proj_xy_smpl_lst=proj_xy_smpl_lst,
                        txyz_dic=txyz_dic,
                        pxyz_dic=pxyz_dic,
                        img=img,
                        motion_weights_vol=motion_weights_vol,
                        chunk=chunk)

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def query_pixel_align_features(self, xy_lst, img_width, img_height, feat_lst):
        num_aug_frames = len(xy_lst)
        query_feat_lst = []
        for i in range(num_aug_frames):
            xy_lst[i][:,:,0] = xy_lst[i][:,:,0] / img_width
            xy_lst[i][:,:,1] = xy_lst[i][:,:,1] / img_height
            xy_lst[i] = xy_lst[i] * 2.0 - 1.0
            uv = xy_lst[i][:, None]
            query_feat = F.grid_sample(
                feat_lst[i],
                uv,
                align_corners=True,
                mode="bilinear",
                padding_mode="zeros",
            ) # [B, C, R*P, 1]

            query_feat_lst.append(query_feat[:, :, 0].permute(0,2,1).contiguous())

        query_feats = torch.stack(query_feat_lst, dim=1) # [B, views, N, feat_dims]

        return query_feats

    def _apply_mlp_kernals(
            self, 
            iter_val,
            pos_flat,
            pts_mask_hard,
            pos_embed_fn,
            cnl_bbox_min_xyz, 
            cnl_bbox_scale_xyz,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            enc_local_feat_lst,
            enc_local_feat_smpl_lst,
            proj_xy_lst,
            proj_xy_smpl_lst,
            txyz_dic,
            pxyz_dic,
            img,
            motion_weights_vol,
            chunk):
        raws = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end].clone()
            local_feat = None
            local_feat_smpl = None
            if cfg.use_enc:
                img_height, img_width, _ = img.shape
                num_aug_frames = len(enc_local_feat_lst)
                query_xy_lst = []
                query_xy_smpl_lst = []
                for i in range(num_aug_frames):
                    query_xy_lst.append(proj_xy_lst[i][:,start:end].clone())
                    if cfg.use_smpl_feat:
                        query_xy_smpl_lst.append(proj_xy_smpl_lst[i].clone())

                query_feat = self.query_pixel_align_features(query_xy_lst, 
                                                    img_width, 
                                                    img_height, 
                                                    enc_local_feat_lst)
                local_feat = query_feat[0].permute(1,0,2).contiguous().to(xyz.device)
                if cfg.use_smpl_feat:
                    query_feat_smpl = self.query_pixel_align_features(query_xy_smpl_lst, 
                                                            img_width, 
                                                            img_height, 
                                                            enc_local_feat_smpl_lst)
    
            if cfg.use_cnl_normalize_coords:
                xyz = (xyz - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0

            if cfg.use_non_rigid_motions and iter_val >= cfg.non_rigid_motion_mlp.kick_in_iter:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem),
                    local_feat=local_feat,
                )
                xyz = result['xyz'].to(cfg.primary_gpus[0])

            if cfg.use_smpl_feat:
                if cfg.use_smpl_early_fusion:
                    query_feat_smpl_fuse = query_feat_smpl.permute(0,2,3,1).contiguous().view(6890, 64*3)
                    assert txyz_dic['txyz'].shape[0]==1
                    xyz_smpl_embed = pos_embed_fn(txyz_dic['txyz'][0])
                    query_feat_smpl_fuse = torch.cat((query_feat_smpl_fuse, xyz_smpl_embed), dim=-1)
                    coord, out_sh, batch_size, grid_coords = get_sparse_conv_input(xyz, txyz_dic, pxyz_dic, cnl_bbox_min_xyz, cnl_bbox_scale_xyz)
                    xyz_sparse = SparseConvTensor(query_feat_smpl_fuse, coord, out_sh, batch_size)
                    feat = self.sparse_conv_net(xyz_sparse, grid_coords)
                    local_feat_smpl = feat.repeat(3,1,1)
                else:
                    local_feat_smpl = []
                    for view in range(query_feat_smpl.shape[1]):
                        coord, out_sh, batch_size, grid_coords = get_sparse_conv_input(xyz, txyz_dic, pxyz_dic, cnl_bbox_min_xyz, cnl_bbox_scale_xyz)
                        xyz_sparse = SparseConvTensor(query_feat_smpl[0,view].view(6890, 64), coord, out_sh, batch_size)
                        feat = self.sparse_conv_net(xyz_sparse, grid_coords)
                        local_feat_smpl.append(feat)
                    local_feat_smpl = torch.stack(local_feat_smpl, dim=1)[0] # [V, C, N]
                local_feat_smpl = local_feat_smpl.permute(2,0,1).contiguous().to(xyz.device)

            # def check_sp_input(xyz, txyz_dic, pxyz_dic, grid_coords):
            #     """
            #     xyz: [6890, 3]
            #     txyz: [6890, 3]
            #     pxyz: [6890, 3]
            #     grid_coords: [1,1,1,6890,3] : query pts
            #     coord: [6890,4] : feature positions
            #     """
            #     # check txyz / pxyz
            #     from wis3d import Wis3D
            #     wis_dir = "/data/jiteng/gdrive"
            #     wis3d = Wis3D(wis_dir, 'figures')

            #     bb = 0
            #     colors = torch.tensor([[255,0,0]]).repeat(6890,1)
            #     wis3d.add_point_cloud(pxyz_dic['pxyz'][bb], colors, name='smpl')
            #     wis3d.add_point_cloud(txyz_dic['txyz'][bb], colors, name='tsmpl')
            #     colors = torch.tensor([[0,255,0]]).repeat(xyz.shape[0],1)
            #     wis3d.add_point_cloud(xyz, colors, name='xyz')

            #     colors = torch.tensor([[255,0,0]]).repeat(6890,1)
            #     plot_coord = coord[bb*6890:(bb+1)*6890, 1:]/torch.tensor(out_sh).cuda() * 2 - 1
            #     wis3d.add_point_cloud(plot_coord[..., [2,1,0]], colors, name='grid_smpl_coord')
            #     colors = torch.tensor([[0,0,255]]).repeat(xyz.shape[0],1)
            #     wis3d.add_point_cloud(grid_coords[bb,0,0], colors, name='grid_coord')
            #     breakpoint()
 
            # check_sp_input(xyz, txyz_dic, pxyz_dic, grid_coords)

            xyz_embedded = pos_embed_fn(xyz)
            raws += [self.cnl_mlp(
                        pos_embed=xyz_embedded,
                        xyz=xyz,
                        local_feat=local_feat, # [N, V, feat_dim]
                        local_feat_smpl=local_feat_smpl, # [N, V, feat_dim]
            )]

        output = {}
        output['raws'] = torch.cat(raws, dim=0).to(cfg.primary_gpus[0])
        if cfg.use_vis_mesh:
            vis_pos = (pos_flat - cnl_bbox_min_xyz[None]) \
                        * cnl_bbox_scale_xyz[None] - 1.0 
            weights = F.grid_sample(input=motion_weights_vol[None][:,:-1], 
                                grid=vis_pos[None, None, None, :, :],           
                                padding_mode='zeros', align_corners=True)
            weights = weights.sum(dim=1)
            weights = weights<0.1
            output['raws'][weights.view(-1)]=0 # include
        else:
            output['raws'][pts_mask_hard.view(-1)]=0 # include

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        if cfg.use_vis_mesh:
            ret = self._render_rays(rays_flat, **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        else:
            for i in range(0, rays_flat.shape[0], cfg.chunk):
                ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
                for k in ret:
                    if k not in all_ret:
                        all_ret[k] = []
                    all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):

        if cfg.use_vis_mesh:
            sigma = F.relu(raw[...,3])
            return sigma

        else:
            def _raw2alpha(raw, dists, act_fn=F.relu):
                return 1.0 - torch.exp(-act_fn(raw)*dists)
    
            sigma = F.relu(raw[...,3])
            dists = z_vals[...,1:] - z_vals[...,:-1]
    
            infinity_dists = torch.Tensor([1e-10])
            infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
            dists = torch.cat([dists, infinity_dists], dim=-1) 
            dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
            rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
            alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
            # if cfg.use_render_mask:
            #     alpha = alpha * raw_mask[:, :, 0]
            #raw_mask[raw_mask>0.03] = 1.0
            #alpha = alpha * raw_mask[:, :, 0]
    
            weights = alpha * torch.cumprod(
                torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                           1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
            rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    
            depth_map = torch.sum(weights * z_vals, -1)
            acc_map = torch.sum(weights, -1)
    
            rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.

            return rgb_map, acc_map, sigma, depth_map


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        
        return results


    @staticmethod
    def _sample_motion_fields_basic(
            tpts,
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            ):
        tpts = tpts.reshape(-1, 3) # [N_rays*N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        pos = (tpts - cnl_bbox_min_xyz[None]) \
                        * cnl_bbox_scale_xyz[None] - 1.0 
        weights = F.grid_sample(input=motion_weights[None], 
                                grid=pos[None, None, None, :, :],           
                                padding_mode='zeros', align_corners=True)
        # weights = SmoothSampler.apply(motion_weights[:, :, :, :, :], 
        #                         pos[:, None, :, :, :].contiguous(), 'zeros', True, False)
        # weights = cu.grid_sample_3d(input=motion_weights[:, :, :, :, :], 
        #                         grid=pos[:, None, :, :, :],           
        #                         padding_mode='zeros', align_corners=True)
        weights = weights[0,:, 0, 0].permute(1, 0).contiguous()
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        weights = weights / weights_sum.clamp(min=0.0001)
        return weights


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            iter_val,
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_max_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None,
            img=None,
            cam_K=None,
            cam_R=None,
            cam_T=None,
            data_enc=None,
            enc_global_feat_lst=None,
            enc_local_feat_lst=None,
            enc_local_feat_smpl_lst=None,
            smpl_vertices=None,
            txyz_dic=None,
            pxyz_dic=None,
            **_):

        if with_batch_dim: 
            cnl_bbox_max_xyz = cnl_bbox_max_xyz[0]
            cnl_bbox_min_xyz = cnl_bbox_min_xyz[0]
            cnl_bbox_scale_xyz = cnl_bbox_scale_xyz[0]
            bgcolor = bgcolor[0]

        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        # torch.manual_seed(0)
        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        # pts = smpl_vertices.unsqueeze(1)
        # pts = pts + torch.randn((pts.shape[0], 1, 6890, 3)).cuda() / 20

        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']
        pts_mask_hard = (pts_mask<0.01)[:,:,0]
        # cnl_pts[pts_mask_hard] = -1e2
        if cfg.use_vis_mesh:
            if cfg.use_cnl_normalize_coords:
                x = torch.arange(-1, 1, 0.005)
                y = torch.arange(-1, 1, 0.005)
                z = torch.arange(-1, 1, 0.015)
                pts = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
                cnl_pts = pts.view(-1, 1, 3).to(cnl_pts.device)
                cnl_pts =  (cnl_pts.to(cnl_bbox_min_xyz.device) + 1) / cnl_bbox_scale_xyz[None, None] + cnl_bbox_min_xyz[None, None]
                # print(torch.min(pts_.view(-1, 3), dim=0), torch.max(pts_.view(-1, 3), dim=0))
            else:
                x = torch.arange(cnl_bbox_min_xyz[0], cnl_bbox_max_xyz[0] + cfg.voxel_size[0],
                            cfg.voxel_size[0]*2)
                y = torch.arange(cnl_bbox_min_xyz[1], cnl_bbox_max_xyz[1] + cfg.voxel_size[1],
                            cfg.voxel_size[1]*2)
                z = torch.arange(cnl_bbox_min_xyz[2], cnl_bbox_max_xyz[2] + cfg.voxel_size[2],
                            cfg.voxel_size[2]*2)
                pts = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
                cnl_pts = pts.view(-1, 1, 3).to(cnl_pts.device)
                # print(torch.min(cnl_pts.view(-1, 3), dim=0), torch.max(cnl_pts.view(-1, 3), dim=0))

        # calculate projected xy coords
        proj_xy_lst = None
        if cfg.use_enc:

            img_lst = []
            cam_K_lst = []
            cam_R_lst = []
            cam_T_lst = []
            dst_Rs_lst = []
            dst_Ts_lst = []
            cnl_gtfms_lst = []
            smpl_vertices_lst = []

            num_data_enc_frames = len(data_enc)
            if cfg.use_data_cross_pose or cfg.use_data_cross_view:
                for aug_idx in range(num_data_enc_frames):
                    img_lst.append(data_enc[aug_idx]['img'])
                    cam_K_lst.append(data_enc[aug_idx]['cam_K'])
                    cam_R_lst.append(data_enc[aug_idx]['cam_R'])
                    cam_T_lst.append(data_enc[aug_idx]['cam_T'])
                    dst_Rs_lst.append(data_enc[aug_idx]['dst_Rs'])
                    dst_Ts_lst.append(data_enc[aug_idx]['dst_Ts'])
                    cnl_gtfms_lst.append(data_enc[aug_idx]['cnl_gtfms'])
                    smpl_vertices_lst.append(data_enc[aug_idx]['smpl_vertices'])

            else:
                # input image
                img_lst.append(img.clone()[None, ...])
                cam_K_lst.append(cam_K[None, ...])
                cam_R_lst.append(cam_R[None, ...])
                cam_T_lst.append(cam_T[None, ...])

            # enc_local_feat_lst = []
            proj_xy_lst = []
            proj_xy_smpl_lst = []
            for aug_idx in range(num_data_enc_frames):
                ################ visualization ################
                # plt.figure()
                # plt.imshow(img_lst[aug_idx][bb].detach().cpu().numpy())
                ################ visualization ################

                img_ = img_lst[aug_idx]
                cam_K_ = cam_K_lst[aug_idx]
                cam_R_ = cam_R_lst[aug_idx]
                cam_T_ = cam_T_lst[aug_idx]
                dst_Rs_ = dst_Rs_lst[aug_idx]
                dst_Ts_ = dst_Ts_lst[aug_idx]
                cnl_gtfms_ = cnl_gtfms_lst[aug_idx]
                smpl_vertices_ = smpl_vertices_lst[aug_idx]

                cam_RT = torch.cat((cam_R_, cam_T_), dim=2).float()
                cam_K_ = cam_K_.float()

                motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs_, 
                                            dst_Ts=dst_Ts_, 
                                            cnl_gtfms=cnl_gtfms_)

                if cfg.use_data_cross_pose:
                    transforms_mat = torch.concat([motion_scale_Rs, motion_Ts[..., None]], dim=3)
                    padding = torch.zeros([motion_scale_Rs.shape[0], 24, 1, 4]).to(motion_Ts.device)
                    padding[..., 3] = 1
                    transforms_mat = torch.concat([transforms_mat, padding], dim=2)
                    transforms_mat_tpose_to_pose = torch.inverse(transforms_mat).contiguous()

                    if cfg.use_vis_mesh:
                        # iterate by trunks
                        bw_lst = []
                        for i in range(0, cnl_pts.shape[0], 1000000):
                            start = i
                            end = i + 1000000
                            bw = self._sample_motion_fields_basic(
                                cnl_pts[start:end],
                                motion_weights_vol,
                                cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
                                )
                            bw_lst.append(bw)
                        bw = torch.cat(bw_lst, dim=0)
                    else:
                        bw = self._sample_motion_fields_basic(
                            cnl_pts,
                            motion_weights_vol,
                            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
                            )
                    orig_shape = list(cnl_pts.shape)

                    if cfg.use_vis_mesh:
                        # iterate by trunks
                        xyz_reproj_lst = []
                        for i in range(0, cnl_pts.shape[0], 1000000):
                            start = i
                            end = i + 1000000
                            xyz_reproj = tpose_points_to_pose_points(cnl_pts.view(-1, 3)[start:end][None], \
                                                            bw[start:end].transpose(0,1)[None], \
                                                            transforms_mat_tpose_to_pose)
                            xyz_reproj_lst.append(xyz_reproj)
                        xyz_reproj = torch.cat(xyz_reproj_lst, dim=1)
                    else:
                        xyz_reproj = tpose_points_to_pose_points(cnl_pts.view(-1, 3)[None], \
                                                            bw.transpose(0,1)[None], \
                                                            transforms_mat_tpose_to_pose)

                    proj_xy = batch_project(xyz_reproj, cam_K_, cam_RT)
                elif cfg.use_data_cross_view:
                    proj_xy = batch_project(pts.view(pts.shape[0], -1, 3), cam_K_, cam_RT)
                else:
                    raise Exception("NOT implemented")

                if cfg.use_smpl_feat:
                    proj_xy_smpl = batch_project(smpl_vertices_, cam_K_, cam_RT)

                ################ visualization ################
                # # proj_xy = batch_project(pxyz_dic['pxyz'][bb][None, ...], cam_K_[bb][None, ...], cam_RT[bb][None, ...])
                # # proj_xy = batch_project(cnl_pts[:, 0], cam_K_[bb][None, ...], cam_RT[bb][None, ...])
                # j_s = proj_xy[bb, :, 0].view(-1)
                # i_s = proj_xy[bb, :, 1].view(-1)
                # plt.plot(j_s.detach().cpu().numpy()[::50], i_s.detach().cpu().numpy()[::50], 'ro')
                # plt.savefig('img{}.jpg'.format(aug_idx))

                # from PIL import Image
                # import numpy as np
                # def load_image(path, to_rgb=True):
                #     img = Image.open(path)
                #     return img.convert('RGB') if to_rgb else img
                # from wis3d import Wis3D
                # wis_dir = "/data/jiteng/gdrive"
                # wis3d = Wis3D(wis_dir, 'figures')
                # wis_img = np.array(load_image('img{}.jpg'.format(aug_idx)))
                # wis3d.add_image(torch.tensor(wis_img).to(torch.uint8), name='img{}'.format(aug_idx))
                ################ visualization ################

                # enc_local_feat_lst.append(enc_local_feat)
                proj_xy_lst.append(proj_xy)
                if cfg.use_smpl_feat:
                    proj_xy_smpl_lst.append(proj_xy_smpl)

        query_result = self._query_mlp(
                                iter_val,
                                pos_xyz=cnl_pts,
                                pts_mask_hard=pts_mask_hard,
                                cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                                cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                                enc_local_feat_lst=enc_local_feat_lst,
                                enc_local_feat_smpl_lst=enc_local_feat_smpl_lst,
                                proj_xy_lst=proj_xy_lst,
                                proj_xy_smpl_lst=proj_xy_smpl_lst,
                                txyz_dic=txyz_dic,
                                pxyz_dic=pxyz_dic,
                                img=img,
                                motion_weights_vol=motion_weights_vol,
                                )
        raw = query_result['raws']

        if cfg.use_vis_mesh:
            sigma = \
                self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)
            
            out = {'alpha' : sigma}
 
        else:
            rgb_map, acc_map, sigma, depth_map = \
                self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)

            out = {'rgb' : rgb_map,  
                'alpha' : acc_map, 
                'depth': depth_map}

            if ('alpha_reg' in cfg.train.lossweights) and cfg.train.lossweights['alpha_reg']>0:
                out['alpha_reg'] = sigma
        return out

    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)


    def data_loader_check(self, **kwargs):

        def min_max_to_bbox_8points(min_coord, max_coord):
            a, b, c = min_coord
            d, e, f = max_coord
            points = torch.zeros((8,3))
            points[0] = torch.tensor([a, b, c])
            points[1] = torch.tensor([d, b, c])
            points[2] = torch.tensor([d, b, f])
            points[3] = torch.tensor([a, b, f])
            points[4] = torch.tensor([a, e, c])
            points[5] = torch.tensor([d, e, c])
            points[6] = torch.tensor([d, e, f])
            points[7] = torch.tensor([a, e, f])
            return points

        from wis3d import Wis3D
        wis_dir = "/data/jiteng/gdrive"
        wis3d = Wis3D(wis_dir, 'figures')
        pbbox = min_max_to_bbox_8points(min_coord=kwargs['dst_bbox']['min_xyz'][0], max_coord=kwargs['dst_bbox']['max_xyz'][0])
        tbbox = min_max_to_bbox_8points(min_coord=kwargs['cnl_bbox_min_xyz'], max_coord=kwargs['cnl_bbox_max_xyz'])

        wis3d.add_boxes(tbbox, name='tbbox', labels='tbbox')
        wis3d.add_boxes(pbbox, name='pbbox', labels='pbbox')

        bb = 0
        colors = torch.tensor([[255,0,0]]).repeat(6890,1)
        wis3d.add_point_cloud(kwargs['pxyz_dic']['pxyz'][bb], colors, name='smpl')
        wis3d.add_point_cloud(kwargs['txyz_dic']['txyz'][bb], colors, name='tsmpl')
        breakpoint()

    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                **kwargs):

        # self.data_loader_check(**kwargs)

        iter_val = kwargs['iter_val']
        if with_batch_dim:
            dst_Rs=dst_Rs
            dst_Ts=dst_Ts
            dst_posevec=dst_posevec
            cnl_gtfms=cnl_gtfms
            motion_weights_priors=motion_weights_priors
        else:
            dst_Rs=dst_Rs[None, ...]
            dst_Ts=dst_Ts[None, ...]
            dst_posevec=dst_posevec[None, ...]
            cnl_gtfms=cnl_gtfms[None, ...]
            motion_weights_priors=motion_weights_priors[None, ...]

#        # correct body pose
#        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0):
#            pose_out = self.pose_decoder(dst_posevec)
#            refined_Rs = pose_out['Rs']
#            refined_Ts = pose_out.get('Ts', None)
#            
#            dst_Rs_no_root = dst_Rs[:, 1:, ...]
#            dst_Rs_no_root = self._multiply_corrected_Rs(
#                                        dst_Rs_no_root, 
#                                        refined_Rs)
#            dst_Rs = torch.cat(
#                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)
#
#            if refined_Ts is not None:
#                dst_Ts = dst_Ts + refined_Ts

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=cfg.non_rigid_motion_mlp.multires,                         
                is_identity=cfg.non_rigid_motion_mlp.i_embed,
                iter_val=iter_val,)

        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input
        })

        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=cnl_gtfms)

        # obtain encoder features
        enc_global_feat_lst = None
        enc_local_feat_lst = None
        if cfg.use_enc:
            data_enc = kwargs['data_enc']
            img_lst = []
            num_data_enc_frames = len(data_enc)
            if cfg.use_data_cross_pose or cfg.use_data_cross_view:
                for aug_idx in range(num_data_enc_frames):
                    img_lst.append(data_enc[aug_idx]['img'])
            else:
                # input image
                img_lst.append(kwargs['img'].clone()[None, ...])

            enc_global_feat_lst = []
            enc_local_feat_lst = []
            enc_local_feat_smpl_lst = []
            for aug_idx in range(num_data_enc_frames):
                img_ = img_lst[aug_idx]
                img_ = img_.permute(0,3,1,2).contiguous()
                img_ = (img_ * 2 ) - 1
                enc_global_feat, enc_local_feat, enc_local_feat_smpl = self.encoder(img_)
                enc_global_feat_lst.append(enc_global_feat)
                enc_local_feat_lst.append(enc_local_feat)
                enc_local_feat_smpl_lst.append(enc_local_feat_smpl)

            kwargs.update({
                'enc_global_feat_lst': enc_global_feat_lst,
                'enc_local_feat_lst': enc_local_feat_lst,
                'enc_local_feat_smpl_lst': enc_local_feat_smpl_lst,
            })

        motion_weights_vol, mweight_delta = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_weights_vol': motion_weights_vol
        })

        if with_batch_dim:
            rays_o, rays_d = rays[0]
            near = near[0]
            far = far[0]
        else:
            rays_o, rays_d = rays
            near = near
            far = far

        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
#        for k in all_ret:
#            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
#            all_ret[k] = torch.reshape(all_ret[k], k_shape)
        all_ret['mweight_delta'] = mweight_delta

        return all_ret
