# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class BaseLine(nn.Module):
    def __init__(self, in_channels=3):
        super(BaseLine, self).__init__()
        self.in_channels = in_channels
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
        self.layers = nn.Sequential(*self.build())

    def build(self):
        in_channels = self.in_channels
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers

    def forward(self, x):
        sources = []
        for k in range(16):
            x = self.layers[k](x)
        sources.append(x)
        for k in range(16, 23):
            x = self.layers[k](x)
        sources.append(x)
        for k in range(23, 35):
            x = self.layers[k](x)
        sources.append(x)
        return x, sources


class Extractor(nn.Module):
    def __init__(self, in_channels=1024):
        super(Extractor, self).__init__()
        self.in_channels = in_channels
        self.cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256, 128, 'S', 256, 128, 256]
        self.layers = nn.Sequential(*self.build())

    def build(self):
        layers = []
        in_channels = self.in_channels
        flag = False
        for k, v in enumerate(self.cfg):
            if in_channels != 'S':
                if v == 'S':
                    conv2d = nn.Conv2d(in_channels, self.cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)
                    layers += [conv2d, nn.BatchNorm2d(self.cfg[k + 1]), nn.ReLU(inplace=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                flag = not flag
            in_channels = v
        return layers

    def forward(self, x):
        sources = []
        for k, v in enumerate(self.layers):
            x = v(x)
            if k % 6 == 3:
                sources.append(x)
        return x, sources


class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        self.in_channles = [256, 512, 1024, 512, 256, 256, 256, 256, 256]
        self.out_channels = [60, 80, 100, 80, 60, 50, 40, 30, 20]
        self.layers = nn.Sequential(*self.build())

    def build(self):
        layers = []
        for i, o in zip(self.in_channles, self.out_channels):
            layers += [nn.Conv2d(i, o, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
        return layers

    def forward(self, sources, bbox):
        sources = [F.relu(net(x)) for net, x in zip(self.layers, sources)]
        res = list()
        for box_index in range(bbox.size(1)):
            box_res = list()
            for source_index in range(len(sources)):
                box_res.append(F.grid_sample(sources[source_index], bbox[:, box_index, :]).squeeze(2).squeeze(2))
            res.append(torch.cat(box_res, 1))
        return torch.stack(res, 1)


class Final(nn.Module):
    def __init__(self):
        super(Final, self).__init__()
        self.cfg = [1040, 512, 256, 128, 64, 1]
        self.layers = nn.Sequential(*self.build())

    def build(self):
        layers = []
        in_channels = self.cfg[0]
        for v in self.cfg[1:-2]:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
        for v in self.cfg[-2:]:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return layers

    def forward(self, x):
        return self.layers(x)


class SST(nn.Module):
    def __init__(self):
        super(SST, self).__init__()
        self.base = BaseLine()
        self.extractor = Extractor()
        self.selector = Selector()
        self.final = Final()
        self.stacker_bn = nn.BatchNorm2d(int(self.final.cfg[0] / 2))
        self.final_dp = nn.Dropout(0.5)

    def forward(self, x_pre, x_next, b_pre, b_next):
        """
        :param x_pre:  b, 3, sst_dim, sst_dim
        :param x_next: b, 3, sst_dim, sst_dim
        :param b_pre:  b, N, 1, 1, 2
        :param b_next: b, N, 1, 1, 2
        :return:
        """
        # forward feature
        feature_pre = self.forward_feature(x_pre, b_pre)
        feature_next = self.forward_feature(x_next, b_next)
        # concate
        feature = self.stacker_features(feature_pre, feature_next)
        # final
        x = self.final_dp(feature)
        x = self.final(x)
        return self.add_unmatched_dim(x)

    def forward_feature(self, x, box):
        sources = list()
        # base
        base, source_base = self.base(x)
        sources += source_base
        # extra
        extra, source_extra = self.extractor(base)
        sources += source_extra
        # selector
        feature = self.selector(sources, box)
        return feature

    def stacker_features(self, features_pre, features_next):
        # TODO thy another way to calculate correlation
        features_pre = features_pre.unsqueeze(2).repeat(1, 1, Config.max_object, 1).permute(0, 3, 1, 2)
        features_next = features_next.unsqueeze(1).repeat(1, Config.max_object, 1, 1).permute(0, 3, 1, 2)
        features_pre = self.stacker_bn(features_pre.contiguous())
        features_next = self.stacker_bn(features_next.contiguous())
        return torch.cat([features_pre, features_next], dim=1)

    def add_unmatched_dim(self, x):
        false_objects_column = torch.ones(x.shape[0], x.shape[1], x.shape[2], 1) * Config.false_constant
        false_objects_row = torch.ones(x.shape[0], x.shape[1], 1, x.shape[3] + 1) * Config.false_constant
        if Config.use_cuda:
            false_objects_column = false_objects_column.cuda()
            false_objects_row = false_objects_row.cuda()
        x = torch.cat([x, false_objects_column], 3)
        x = torch.cat([x, false_objects_row], 2)
        return x

    def get_similarity(self, feature1, feature2):
        # forward
        feature = self.stacker_features(feature1, feature2)
        x = self.final_dp(feature)
        x = self.final(x)
        x = self.add_unmatched_dim(x)[0, 0, :]
        # softmax and select
        x_pre = F.softmax(x, dim=1)
        x_next = F.softmax(x, dim=0)
        res = (x_pre + x_next)[:-1, :-1] / 2
        res = torch.cat([res, x_pre[:-1, -1:]], 1)
        return res


def build_sst(size=900):
    if size != 900:
        print('Error: Sorry only SST 900 is supported currently!')
        return
    # from . import sst_custom
    # return sst_custom.SST()
    return SST()
