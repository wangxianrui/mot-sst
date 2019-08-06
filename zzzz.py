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

max_interval = 25
track_data = np.loadtxt('result/test/txt/sequence_02.txt', delimiter=',')
track_group = pd.DataFrame(track_data).groupby(1)
track_ids = track_group.indices.keys()
video_fragment = []
start = time.time()
for id in track_ids:
    track = track_group.get_group(id).values
    fragment = [np.min(track[:, 0]), np.max(track[:, 0])]
    if len(video_fragment) == 0 or fragment[0] - video_fragment[-1][1] > max_interval:
        video_fragment.append(fragment)
    else:
        last_fragment = video_fragment[-1]
        video_fragment[-1] = [min(last_fragment[0], fragment[0]), max(last_fragment[1], fragment[1])]
with open('test', 'a') as file:
    file.write('sequence_01\n')
    for fragment in video_fragment:
        file.write('\t' + str(fragment) + '\n')
