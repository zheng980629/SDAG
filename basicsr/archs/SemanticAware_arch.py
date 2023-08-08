import torch
from torch import nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
import os
from math import exp

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer, ConvLReLUNoBN, upsample_and_concat
from basicsr.utils.registry import ARCH_REGISTRY
from torchvision import models


class Vgg16_first3(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16_first3, self).__init__()
        vgg = models.vgg16(pretrained=False)
        vgg.load_state_dict(torch.load('/gdata1/zhengns/vgg16-397923af.pth'))
        vgg.eval()
        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(3, 7):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(7, 12):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(12, 21):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # self.id = id
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        # h_relu2 = self.slice2(h_relu1)
        # h_relu3 = self.slice3(h_relu2)
        # h_relu4 = self.slice4(h_relu3)
        # h_relu5 = self.slice5(h_relu4)
        # out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return h_relu1


class HIN(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HIN, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out_1 = self.norm(out_1)
            # feature_save(out_1,'IN')
            # feature_save(out_2,'ID')
            out = torch.cat([out_1, out_2], dim=1)

        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'HIN':
            return HIN(channel_in, channel_out)
        else:
            return None

    return constructor



class InvBlock(nn.Module):
    def __init__(self,channel_num, channel_split_num, subnet_constructor=subnet('HIN'), clamp=0.8):   ################  split_channel一般设为channel_num的一半
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)


    def forward(self, x):
        # if not rev:
        # invert1x1conv
        # x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out



@ARCH_REGISTRY.register()
class SemanticAwareNet(nn.Module):
    def __init__(self, channels):
        super(SemanticAwareNet,self).__init__()

        self.vgg_extractor = Vgg16_first3()

        self.process1 = InvBlock(channels, channels//2)
        self.process2 = InvBlock(channels, channels // 2)
        self.process3 = InvBlock(channels, channels // 2)

        self.ConvOut = nn.Conv2d(channels, 3, 1, 1, 0)

    def forward(self, x):
        # feature_save(x, '11')

        x = self.vgg_extractor(x)

        x1 = self.process1(x)
        x2 = self.process2(x1)
        x3 = self.process3(x2)

        out = self.ConvOut(x3)

        return out