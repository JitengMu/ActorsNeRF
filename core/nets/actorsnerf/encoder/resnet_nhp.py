"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.autograd.profiler as profiler
import functools
import numpy as np
import time


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError(
            "normalization layer [%s] is not found" % norm_type)
    return norm_layer


def make_encoder(conf, **kwargs):
    enc_type = conf.get_string("type", "spatial")  # spatial | global
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net


class resnet18(nn.Module):


    def __init__(
            self,
            backbone="resnet18",
            pretrained=True,
            num_layers=3,  # 2: 128 3: 256 4: 512
            index_interp="bilinear",
            index_padding="zeros",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            norm_type="batch",
    ):


        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = get_norm_layer(norm_type)

        pretrained = 'True'
        print("Using torchvision", backbone, "encoder")
        print('Pretrained: ' + str(pretrained))
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained, norm_layer=norm_layer
        )


        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        img_feat_size = 256
        embed_size = 64
        if img_feat_size == 128:
            num_layers = 2
        elif img_feat_size == 256:
            num_layers = 3
        elif img_feat_size == 512:
            num_layers = 4
        num_layers = 3


        self.reduction_layer = nn.Conv2d(img_feat_size, embed_size,1)
        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, res, arch):

        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )


        pixel_feat_map = None
        holder_feat_map = None

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:


            x = self.model.conv1(x)

            x = self.model.bn1(x)

            x = self.model.relu(x)

            latents = [x]


            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            x2 = self.model.layer2(x)
            if self.num_layers > 2:
                latents.append(x2)
            x3 = self.model.layer3(x2)
            if self.num_layers > 3:
                latents.append(x3)
            x4 = self.model.layer4(x3)
            if self.num_layers > 4:
                latents.append(x4)

            out = self.avgpool(x4)
            global_feat = torch.flatten(out, 1)

            align_corners = None if self.index_interp == "nearest " else True


            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )

            pixel_feat_map = torch.cat(latents, dim=1)

        holder_feat_map = self.reduction_layer(pixel_feat_map)

        return global_feat, pixel_feat_map, holder_feat_map


    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )
