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

data = pd.read_csv('result/test/txt/sequence_01.txt')
print(data)
