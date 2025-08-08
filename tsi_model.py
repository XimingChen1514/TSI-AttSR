import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN
from torchvision.models import vgg19
import time
import matplotlib.pyplot as plt
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out) 
        return out

def swish(x):
    return F.relu(x)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)

class GGCA(nn.Module):
    def __init__(self, channel, h, w, reduction=16, num_groups=4):
        super(GGCA, self).__init__()
        self.num_groups = num_groups
        self.group_channels = channel // num_groups
        self.h = h
        self.w = w

        # H
        self.avg_pool_h = nn.AdaptiveAvgPool2d((h, 1))
        self.max_pool_h = nn.AdaptiveMaxPool2d((h, 1))
        # W
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, w))
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, w))

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.group_channels, out_channels=self.group_channels // reduction,
                      kernel_size=(1, 1)),
            nn.BatchNorm2d(self.group_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.group_channels // reduction, out_channels=self.group_channels,
                      kernel_size=(1, 1))
        )
        # Sigmoid
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        assert channel % self.num_groups == 0, "The number of channels must be divisible by the number of groups."

        x = x.view(batch_size, self.num_groups, self.group_channels, height, width)

        x_h_avg = self.avg_pool_h(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, self.h, 1)
        x_h_max = self.max_pool_h(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, self.h, 1)

        x_w_avg = self.avg_pool_w(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, self.w)
        x_w_max = self.max_pool_w(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, self.w)

        y_h_avg = self.shared_conv(x_h_avg.view(batch_size * self.num_groups, self.group_channels, self.h, 1))
        y_h_max = self.shared_conv(x_h_max.view(batch_size * self.num_groups, self.group_channels, self.h, 1))

        y_w_avg = self.shared_conv(x_w_avg.view(batch_size * self.num_groups, self.group_channels, 1, self.w))
        y_w_max = self.shared_conv(x_w_max.view(batch_size * self.num_groups, self.group_channels, 1, self.w))

        att_h = self.sigmoid_h(y_h_avg + y_h_max).view(batch_size, self.num_groups, self.group_channels, self.h, 1)
        att_w = self.sigmoid_w(y_w_avg + y_w_max).view(batch_size, self.num_groups, self.group_channels, 1, self.w)

        out = x * att_h * att_w
        out = out.view(batch_size, channel, height, width)

        return out

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)

        self.resblock2 = ResidualBlock(64)
        self.resblock3 = ResidualBlock(64)
        self.resblock4 = ResidualBlock(64)

        self.ggca = GGCA(64,96,96)

        self.convupsample1 = nn.Conv2d(256, 64, 3, stride=1, padding=1)

        self.dconv1 = DCN(64, 64, 3, 1, 1)
        self.dconv2 = DCN(64, 64, 3, 1, 1)
        self.dconv3 = DCN(64, 64, 3, 1, 1)

        self.conv2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

    def forward(self, x, y):
        x1 = swish(self.conv1(x))
        y_norm = self.norm(y)
        y1 = torch.sigmoid(y_norm)
        x2 = x1 * y1 * 1

        x3 = self.resblock2(x2)
        x4 = self.resblock3(x3)
        x5 = self.resblock4(x4)

        a5 = self.ggca(x5)
        a4 = self.ggca(x4)
        a3 = self.ggca(x3)
        a2 = self.ggca(x2)

        b = torch.cat((a5, a4), dim=1)
        b1 = torch.cat((b, a3), dim=1)
        b2 = torch.cat((b1, a2), dim=1)

        b3 = self.convupsample1(b2)

        z1 = swish(self.dconv1(b3))
        z2 = swish(self.dconv2(z1))
        z3 = swish(self.dconv3(z2))

        z4 = self.conv2(z3)
        z5 = torch.tanh(z4)
        z5 = z5 + x

        return z5



