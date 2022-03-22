import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


def get_flat_mask(img, kernel_size=7, std_thresh=0.03, scale=1):
    img = F.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
    B, _, H, W = img.size()
    r, g, b = torch.unbind(img, dim=1)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(dim=1)
    l_img_pad = F.pad(l_img, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
    unf_img = F.unfold(l_img_pad, kernel_size=kernel_size, padding=0, stride=1)
    std_map = torch.std(unf_img, dim=1, keepdim=True).view(B, 1, H, W)
    mask = torch.lt(std_map, std_thresh).float()

    return mask


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


## add SELayer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## add SEResBlock
class SEResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(SEResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(SELayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x

        return res


if __name__ == '__main__':
    import cv2
    import numpy as np
    img = cv2.imread('/data/qilu/projects/vincent/downloads/DIV2K_val_3cases/0808x4.png').astype(np.float32) / 255
    img = np.ascontiguousarray(np.transpose(img[:, :, ::-1], (2, 0, 1)))
    img = torch.from_numpy(img).float().unsqueeze(0)
    mask = get_flat_mask(img, kernel_size=11, std_thresh=0.025, scale=4)
    img = F.interpolate(img, scale_factor=4, mode='bicubic')
    img = mask * img
    img = (img.squeeze().clamp(0, 1).numpy() * 255).round()
    img = np.transpose(img[::-1, :, :], (1, 2, 0)).astype(np.uint8)
    cv2.imwrite('flat_0808_lr_v2.png', img)
