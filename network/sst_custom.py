# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""
import torch
import torchvision.models as th_models
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class BaseLine(nn.Module):
    def __init__(self, in_channels=3):
        super(BaseLine, self).__init__()
        self.in_channels = in_channels
        temp_model = th_models.resnet50()
        self.layers = nn.ModuleList(temp_model.children())[:-3]

    def forward(self, x):
        sources = list()
        for i, m in enumerate(self.layers):
            x = m(x)
            if i in [4, 5, 6]:
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
            layers += [
                nn.Sequential(
                    nn.Conv2d(i, o, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(o),
                    nn.ReLU()
                )]
        return layers

    def forward(self, sources, bbox):
        sources = [net(x) for net, x in zip(self.layers, sources)]
        res = list()
        for box_index in range(bbox.size(1)):
            box_res = list()
            for source_index in range(len(sources)):
                box_res.append(F.grid_sample(sources[source_index], bbox[:, box_index, :]).squeeze(2).squeeze(2))
            res.append(torch.cat(box_res, 1))
        return torch.stack(res, 1)


class SST(nn.Module):
    def __init__(self):
        super(SST, self).__init__()
        self.base = BaseLine()
        self.extractor = Extractor()
        self.selector = Selector()

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
        return self.add_unmatched_dim(feature)

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
        features_next = features_next.permute(0, 2, 1)
        product = torch.matmul(features_pre, features_next)
        pre_dis = torch.sqrt(torch.sum(torch.pow(features_pre, 2), 2)).reshape(-1, features_pre.shape[1], 1)
        next_dis = torch.sqrt(torch.sum(torch.pow(features_next, 2), 1)).reshape(-1, 1, features_next.shape[2])
        total_dis = pre_dis * next_dis
        assert product.shape == total_dis.shape
        feature = product / total_dis
        return feature.unsqueeze(dim=0)

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
        x = self.add_unmatched_dim(feature)
        # softmax and select
        x_pre = F.softmax(x, dim=1)
        x_next = F.softmax(x, dim=0)
        res = (x_pre + x_next)[:-1, :-1] / 2
        res = torch.cat([res, x_pre[:-1, -1:]], 1)
        return res
