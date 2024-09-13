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

import torch.nn as nn
import torch.nn.functional as F
import torch

from configs import cfg


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        if cfg.use_smpl_early_fusion:
            self.conv0 = double_conv(64*3+63, 64, 'subm0')
        else:
            self.conv0 = double_conv(64, 64, 'subm0')
        self.down0 = stride_conv(64, 64, 'down0')

        self.conv1 = double_conv(64, 64, 'subm1')
        self.down1 = stride_conv(64, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        # self.down3 = stride_conv(128, 128, 'down3')

        # self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x, grid_coords):

        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        feature_1 = F.grid_sample(net1,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)

        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        feature_2 = F.grid_sample(net2,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        feature_3 = F.grid_sample(net3,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        # net = self.down3(net)

        # net = self.conv4(net)
        # net4 = net.dense()
        # feature_4 = F.grid_sample(net4,
        #                           grid_coords,
        #                           padding_mode='zeros',
        #                           align_corners=True)
        '''

        '''

        # features = torch.cat((feature_1, feature_2, feature_3, feature_4),
        #                      dim=1)
        features = torch.cat((feature_1, feature_2, feature_3), dim=1)
        features = features.view(features.size(0), -1, features.size(4))

        return features


def single_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   1,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SparseConv3d(in_channels,
                     out_channels,
                     3,
                     2,
                     padding=1,
                     bias=False,
                     indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
