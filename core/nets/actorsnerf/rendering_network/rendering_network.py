import torch.nn as nn
import torch.nn.functional as F
import torch


#from lib.config import cfg
#from lib.networks.encoder import SpatialEncoder
import math
import time
from configs import cfg

def combine_interleaved(t, num_input=4, agg_type="average"):

    # t = t.reshape(-1, num_input, *t.shape[1:])
    assert len(t.shape)==3

    if agg_type == "average":
        t = torch.mean(t, dim=1)

    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


class CanonicalMLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, 
                 input_ch=3, skips=None,
                 **_):
        super(CanonicalMLP, self).__init__()

        self.actvn = nn.ReLU()
        img_feat_size = 64+64+128

        if cfg.use_smpl_feat and cfg.use_pixel_feat:
            self.smpl_fc = nn.Linear(256, 256)
            self.pixel_fc = nn.Linear(img_feat_size, 256)
            self.alpha_fc_0 = nn.Linear(512, 256)
            self.rgb_fc_0 = nn.Linear(512, 256)
        elif cfg.use_smpl_feat and (not cfg.use_pixel_feat):
            self.smpl_fc = nn.Linear(256, 256)
            self.alpha_fc_0 = nn.Linear(256, 256)
            self.rgb_fc_0 = nn.Linear(256, 256)
        elif (not cfg.use_smpl_feat) and cfg.use_pixel_feat:
            self.pixel_fc = nn.Linear(img_feat_size, 256)
            self.alpha_fc_0 = nn.Linear(256, 256)
            self.rgb_fc_0 = nn.Linear(256, 256)
        else:
            self.alpha_fc_0 = nn.Linear(63, 256)
            self.rgb_fc_0 = nn.Linear(63, 256)

        self.alpha_fc_1 = nn.Linear(256 + 63, 256)
        self.alpha_fc_2 = nn.Linear(256, 256)
        self.alpha_fc_3 = nn.Linear(256, 256)
        self.alpha_fc = nn.Linear(256, 1)

        self.rgb_fc_1 = nn.Linear(256 + 63, 256)
        self.rgb_fc_2 = nn.Linear(256, 256)
        self.rgb_fc_3 = nn.Linear(256, 256)
        self.rgb_fc = nn.Linear(256, 3)


    def forward(self, pos_embed, xyz, local_feat, local_feat_smpl, **_):
        """
            :pos_embed: (N/B, 63)
            :xyz: (N/B, 3)
            :local_feat: (N/B, V, 256)
            :local_feat_smpl: (N/B, V, 384)
        """
        num_views = 3
        if cfg.use_smpl_feat and cfg.use_pixel_feat:
            smpl_feat_input = self.actvn(self.smpl_fc(local_feat_smpl))
            pixel_feat_input = self.actvn(self.pixel_fc(local_feat))
            concat_feat = torch.cat((smpl_feat_input, pixel_feat_input), dim=-1)
        elif (not cfg.use_smpl_feat) and cfg.use_pixel_feat:
            pixel_feat_input = self.actvn(self.pixel_fc(local_feat))
            concat_feat = pixel_feat_input
        elif cfg.use_smpl_feat and (not cfg.use_pixel_feat):
            smpl_feat_input = self.actvn(self.smpl_fc(local_feat_smpl))
            concat_feat = smpl_feat_input
        else:
            concat_feat = pos_embed[:, None].repeat(1,num_views,1)

        net_alpha = self.actvn(self.alpha_fc_0(concat_feat))
        net_alpha = torch.concat((net_alpha, pos_embed[:, None].repeat(1,num_views,1)), dim=-1)
        net_alpha = self.alpha_fc_1(net_alpha)

        net_alpha = combine_interleaved(
            net_alpha, num_views, "average"
        )
        net_alpha = self.actvn(net_alpha)

        net_alpha = self.actvn(self.alpha_fc_2(net_alpha))
        net_alpha = self.actvn(self.alpha_fc_3(net_alpha))
        alpha = self.alpha_fc(net_alpha)

        net_rgb = self.actvn(self.rgb_fc_0(concat_feat))
        net_rgb = torch.concat((net_rgb, pos_embed[:, None].repeat(1,num_views,1)), dim=-1)
        net_rgb = self.actvn(self.rgb_fc_1(net_rgb))
        net_rgb = self.rgb_fc_2(net_rgb)
        # net_rgb = torch.cat((net_rgb, self.rgb_res_0(pixel_feat)), dim=-1)

        net_rgb = combine_interleaved(
            net_rgb, num_views, "average"
        )
        net_rgb = self.actvn(net_rgb)

        # net_rgb = net_rgb + self.rgb_res_1(pixel_feat)

        net_rgb = self.actvn(self.rgb_fc_3(net_rgb))
        rgb = self.rgb_fc(net_rgb)

        raw = torch.cat((rgb, alpha), dim=1)

        return raw