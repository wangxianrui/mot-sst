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
from scipy.optimize import linear_sum_assignment

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

# data = np.random.randn(5)
# print(data)
# mask = data > 0
# print(mask)
# print(data[mask])


cost = np.random.rand(4, 8)
print(cost)
row, col = linear_sum_assignment(cost)
print(col)
