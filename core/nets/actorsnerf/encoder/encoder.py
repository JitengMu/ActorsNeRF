import torch
import torch.nn as nn
from .resnet_nhp import *

from configs import cfg

class Encoder(nn.Module):
    def __init__(self, encoder_arch):
        super(Encoder, self).__init__()
        self.encoder_arch = encoder_arch
        if self.encoder_arch=='resnet_18' or self.encoder_arch=='resnet_18_concat':
            self.model = resnet18(pretrained=True)
        elif self.encoder_arch=='resnet_50' or self.encoder_arch=='resnet_50_concat':
            self.model = resnet50(pretrained=True)
        elif self.encoder_arch=='vfs_resnet_18':
            self.model = resnet18(pretrained=False)
            ckpt_pth = "data/r18_nc_sgd_cos_100e_r2_1xNx8_k400-db1a4c0d.pth"
            ckpt = self.__load_vfs_model(ckpt_pth, self.model)
        elif self.encoder_arch=='vfs_resnet_50':
            self.model = resnet50(pretrained=False)
            ckpt_pth = "data/r50_nc_sgd_cos_100e_r5_1xNx2_k400-d7ce3ad0.pth"
            ckpt = self.__load_vfs_model(ckpt_pth, self.model)
        elif self.encoder_arch=='moco_resnet_50':
            self.model = resnet50(pretrained=False)
            ckpt_pth = "data/moco_v2_800ep_pretrain.pth.tar"
            ckpt = self.__load_moco_model(ckpt_pth, self.model)
        self.res = 512

        train_param_lst = ['reduction_layer']
        if cfg.train_encoder==False:
            for n, param in self.model.named_parameters():
                for train_param in train_param_lst:
                    if train_param in n:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        # for param in self.model.parameters():
        #         param.requires_grad = False
        # self.model.eval()

    def forward(self, x):
        x = self.model(x, self.res, self.encoder_arch)
        return x
