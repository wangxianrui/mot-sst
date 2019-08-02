# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import cv2
import os
import shutil

# x1 = torch.rand(45, 512, requires_grad=True)
# x2 = torch.rand(45, 512, requires_grad=True).transpose(1, 0)
# y = torch.matmul(x1, x2)
# x1_dis = torch.sqrt(torch.sum(torch.pow(x1, 2), 1).reshape(-1, 1))
# x2_dis = torch.sqrt(torch.sum(torch.pow(x2, 2), 0).reshape(1, -1))
# total_dis = x1_dis * x2_dis
# assert y.shape == total_dis.shape
# similarity = y / (x1_dis * x2_dis)
# print(similarity)
# res = torch.sum(similarity)
# res.backward()

data_dir = '../dataset/CAR_tracking'
index = 1
for name in os.listdir(data_dir):
    print(os.path.join(data_dir, 'sequence_{0:03}'.format(index)))
    shutil.move(os.path.join(data_dir, name), os.path.join(data_dir, 'sequence_{0:03}'.format(index)))
    index += 1
