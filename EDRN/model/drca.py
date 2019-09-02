## This code is built on EDSR and RCAN

import sys
sys.path.append('/home/dongbin/DL/EDRN/')

from model import common

import torch.nn as nn
import torch


def make_model(args, parent=False):
    return DRCA(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
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

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=False, bn=True, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Channel Attention Block with Projection (RCABP)
class RCABP(nn.Module):
    def __init__(
            self, conv, in_feat, n_feat, kernel_size, reduction,
            bias=False, bn=True, act=nn.ReLU(True), res_scale=1):

        super(RCABP, self).__init__()
        modules_body = []
        for i in range(2):
            if i == 0:
                # modules_body.append(conv(in_feat, 4*n_feat, 1, bias=bias))
                # modules_body.append(act)
                modules_body.append(conv(in_feat, n_feat, kernel_size, bias=bias))
            else:
                modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        projection_body = []
        projection_body.append(conv(in_feat, n_feat, 1, bias=bias))
        self.body = nn.Sequential(*modules_body)
        self.proj = nn.Sequential(*projection_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += self.proj(x)
        return res

## Residual Group
class ResidualGroup(nn.Module):
    def __init__(self, conv, in_feat, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body.append(
            RCABP(conv, in_feat, n_feat, kernel_size, reduction, bias=True, bn=True, act=nn.ReLU(True), res_scale=1))
        for i in range(n_resblocks - 1):
            modules_body.append(
                RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=True, act=nn.ReLU(True), res_scale=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return torch.cat([x, res], 1)

## DenseNet with Residual Channel Attention blocks (DRCA)
class DRCA(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DRCA, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = []
        for i in range(n_resgroups):
            modules_body.append(ResidualGroup(conv, (i + 1) * n_feats, n_feats, kernel_size, reduction, act=act,
                                              res_scale=args.res_scale, n_resblocks=n_resblocks))

        modules_body.append(conv((n_resgroups + 1) * n_feats, n_feats, 1))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))