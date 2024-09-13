import os

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from third_parties.lpips import LPIPS

from core.train import create_lr_updater
from core.data import create_dataloader
from core.utils.network_util import set_requires_grad
from core.utils.train_util import cpu_data_to_gpu, Timer
from core.utils.image_util import tile_images, to_8b_image

from configs import cfg

import wandb

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'sub_idx', 'cam_idx', 'img_width', 'img_height']

def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


class Trainer(object):
    def __init__(self, network, optimizer):
        print('\n********** Init Trainer ***********')

        # network = network.cuda().deploy_mlps_to_secondary_gpus()
        self.network = network

        self.optimizer = optimizer
        self.update_lr = create_lr_updater()

        if cfg.resume and Trainer.ckpt_exists(cfg.load_net):
            self.load_ckpt(f'{cfg.load_net}')
        elif os.path.exists(cfg.load_cate_net):
            self.load_cate_ckpt(f'{cfg.load_cate_net}')
            self.iter = 100000
            print("################ load category level ckpts ###################")
        else:
            self.iter = 0
            self.save_ckpt('init')
            self.iter = 1

        network = network.cuda().deploy_mlps_to_secondary_gpus()

        self.timer = Timer()

        if "lpips" in cfg.train.lossweights.keys():
            self.lpips = LPIPS(net='vgg')
            set_requires_grad(self.lpips, requires_grad=False)
            self.lpips = nn.DataParallel(self.lpips).cuda()

        print("Load Progress Dataset ...")
        self.prog_dataloader = create_dataloader(data_type='progress', dataset_mode='cat_level')

        print('************************************')
        if cfg.use_wandb:
            # wandb setup
            wandb_config = dict(
                    pose_correct = cfg.use_pose_correction,
                    non_rigid = cfg.use_non_rigid_motions,
                    cnl_norm_coords = cfg.use_cnl_normalize_coords,
                    train_encoder = cfg.train_encoder,
                    use_data_cross_pose = cfg.use_data_cross_pose,
                    use_data_cross_view = cfg.use_data_cross_view,
                    use_smpl_feat = cfg.use_smpl_feat,
                    use_pixel_feat = cfg.use_pixel_feat,
                    )
            if cfg.use_wandb:
                wandb.init(
                        project="actorsnerf",
                        name=cfg.experiment,
                        config=wandb_config,
                        )

    @staticmethod
    def get_ckpt_path(name):
        return os.path.join(cfg.logdir, f'{name}.tar')

    @staticmethod
    def ckpt_exists(name):
        return os.path.exists(Trainer.get_ckpt_path(name))

    ######################################################3
    ## Training 

    def get_img_rebuild_loss(self, loss_names, rgb, target, net_output):
        losses = {}

        if "mse" in loss_names:
            losses["mse"] = img2mse(rgb, target)

        if "l1" in loss_names:
            losses["l1"] = img2l1(rgb, target)

        if "lpips" in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses["lpips"] = torch.mean(lpips_loss)

        if "mweight_l1" in loss_names:
            mweight_l1 = torch.norm(net_output['mweight_delta'], dim=1, p=1)
            losses['mweight_l1'] = torch.mean(mweight_l1.view(-1))

        return losses

    def get_loss(self, net_output, 
                 patch_masks, bgcolor, targets, div_indices):

        lossweights = cfg.train.lossweights
        loss_names = list(lossweights.keys())

        rgb = net_output['rgb']
        losses = self.get_img_rebuild_loss(
                        loss_names, 
                        _unpack_imgs(rgb, patch_masks, bgcolor,
                                     targets, div_indices), 
                        targets, net_output)

        train_losses = [
            weight * losses[k] for k, weight in lossweights.items()
        ]

        if cfg.use_wandb:
            wandb_dic = {}
            for i in range(len(loss_names)):
                wandb_dic[loss_names[i]] = train_losses[i]
            wandb.log(wandb_dic)

        return sum(train_losses), \
               {loss_names[i]: train_losses[i] for i in range(len(loss_names))}

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1

        self.network.train()
        cfg.perturb = cfg.train.perturb

    def train_end(self):
        pass

    def train(self, epoch, train_dataloader):
        self.train_begin(train_dataloader=train_dataloader)

        self.timer.begin()
        for batch_idx, batch in enumerate(train_dataloader):
            if self.iter > cfg.train.maxiter:
                break

            self.optimizer.zero_grad()

            # only access the first batch as we process one image one time
            batch = batch[0]
            for k, v in batch.items():
                if k=="data_enc" or k=='txyz_dic' or k=='pxyz_dic' or k=='dst_bbox':
                    batch[k] = v
                else:
                    batch[k] = v[0]

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            net_output = self.network(**data)

            train_loss, loss_dict = self.get_loss(
                net_output=net_output,
                patch_masks=data['patch_masks'],
                bgcolor=data['bgcolor'] / 255.,
                targets=data['target_patches'],
                div_indices=data['patch_div_indices'])

            train_loss.backward()
            self.optimizer.step()

            if self.iter % cfg.train.log_interval == 0:
                loss_str = f"Loss: {train_loss.item():.4f} ["
                for k, v in loss_dict.items():
                    loss_str += f"{k}: {v.item():.4f} "
                loss_str += "]"

                log_str = 'Epoch: {} [Iter {}, {}/{} ({:.0f}%), {}] {}'
                log_str = log_str.format(
                    epoch, self.iter,
                    batch_idx * cfg.train.batch_size, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), 
                    self.timer.log(),
                    loss_str)
                print(log_str)

            is_reload_model = False
            if self.iter in [100, 300, 1000, 2500] or \
                self.iter % cfg.progress.dump_interval == 0:
                is_reload_model = self.progress()

            if not is_reload_model:
                if self.iter % cfg.train.save_checkpt_interval == 0:
                    self.save_ckpt('latest')

                    if cfg.use_wandb and (self.iter % 20000==0):
                        s3_ckpt = Trainer.get_ckpt_path('latest')
                        s3_ckpt_ = os.path.join(f'{cfg.experiment}/checkpoints/latest.tar')
                        os.system("aws s3 cp {} s3://humannerf/{}".format(s3_ckpt, s3_ckpt_))     

                if cfg.save_all:
                    if self.iter % cfg.train.save_model_interval == 0:
                        self.save_ckpt(f'iter_{self.iter}')

                        if cfg.use_wandb and (self.iter % 10000==0):
                            s3_ckpt = Trainer.get_ckpt_path(f'iter_{self.iter}')
                            s3_ckpt_ = os.path.join(f'{cfg.experiment}/checkpoints/iter_{self.iter}.tar')
                            os.system("aws s3 cp {} s3://humannerf/{}".format(s3_ckpt, s3_ckpt_))     

                self.update_lr(self.optimizer, self.iter)

                self.iter += 1
    
    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        cfg.perturb = 0.

    def progress_end(self):
        self.network.train()
        cfg.perturb = cfg.train.perturb

    def progress(self):
        self.progress_begin()

        print('Evaluate Progress Images ...')

        images = []
        is_empty_img = False
        for _, batch in enumerate(tqdm(self.prog_dataloader)):

            # only access the first batch as we process one image one time
            batch = batch[0]
            for k, v in batch.items():
                if k=="data_enc" or k=='txyz_dic' or k=='pxyz_dic' or k=='dst_bbox':
                    batch[k] = v
                else:
                    batch[k] = v[0]

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.network(**data)

            rgb = net_output['rgb'].data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs']

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs

            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            images.append(np.concatenate([rendered, truth], axis=1))

             # check if we create empty images (only at the begining of training)
            if self.iter <= 5000 and \
                np.allclose(rendered, np.array(cfg.bgcolor), atol=5.):
                is_empty_img = True
                break

        tiled_image = tile_images(images)

        if cfg.use_wandb:
            wandb_img = wandb.Image(tiled_image, caption="Image")
            wandb.log({"images": wandb_img})

        Image.fromarray(tiled_image).save(
            os.path.join(cfg.logdir, "prog_{:06}.jpg".format(self.iter)))

        if is_empty_img:
            print("Produce empty images; reload the init model.")
            self.load_ckpt('init')
            
        self.progress_end()

        return is_empty_img


    ######################################################3
    ## Utils

    def save_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Save checkpoint to {path} ...")

        torch.save({
            'iter': self.iter,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path, map_location='cuda:0')
        self.iter = ckpt['iter'] + 1

        self.network.load_state_dict(ckpt['network'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def load_cate_ckpt(self, path):
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path, map_location='cuda:0')
        self.iter = 0

        # model_dict_filter = self.network.state_dict().copy()
        # # exclude_list = ['mweight_vol_decoder', 'non_rigid_mlp', 
        # #                 'xyzc_net', 'pixel_fc', 'smpl_fc', 'alpha_fc', 'rgb_fc', 'rgb_res',
        # #                 'encoder', 'fuse_net']
        exclude_list = cfg.exclude_list
        # # for k, v in self.network.state_dict().items():
        # #     if k in exclude_list:
        # #         del model_dict_filter[k]
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in ckpt['network'].items() if k in model_dict_filter}
        # # 2. overwrite entries in the existing state dict
        # model_dict_filter.update(pretrained_dict) 
        # # 3. load the new state dict
        # self.network.load_state_dict(model_dict_filter, strict=True)
        self.network.load_state_dict(ckpt['network'], strict=True)
        
        # fix model weights
        for n, p in self.network.named_parameters():
            for m in exclude_list:
                if m in n:
                    p.requires_grad = False
            print(n, p.requires_grad)

        # self.optimizer.load_state_dict(ckpt['optimizer'])
        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()
