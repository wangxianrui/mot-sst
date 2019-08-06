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
import numpy as np
import pandas as pd
import torchvision.models as models
import cv2
import os
import shutil
from scipy.optimize import linear_sum_assignment
import time

a = torch.rand(1, 520, 45, 1)
b = torch.rand(1, 520, 1, 45)
c = torch.matmul(a, b)
print(c.shape)
