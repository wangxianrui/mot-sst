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
from config import EvalConfig as Config
from torch.utils.data import DataLoader
from pipline.mot_train_dataset_ import MOTTrainDataset
import random

video_dir = 'D:/movies_08_06/SG教会我爱你第一季-01'
detection = np.loadtxt(os.path.join(video_dir, 'det', 'det.txt'), delimiter=',')
detection = detection[detection[:, 6] > Config.high_confidence, :]
frame_index = np.unique(detection[:, 0]).astype(np.int)
cv2.namedWindow('win', cv2.WINDOW_NORMAL)
for index in frame_index:
    img = cv2.imdecode(np.fromfile(os.path.join(video_dir, 'img1', '{0:06}.jpg'.format(index)), dtype=np.uint8), -1)
    frame_det = detection[detection[:, 0] == index, :]
    print(index, '\t', len(frame_det))
    for det in frame_det:
        x, y, w, h = [int(n) for n in det[2:6]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imshow('win', img)
    cv2.waitKey()
