# -*- coding: <encoding name> -*-
"""
@authors: rayenwang
@time: 2019/7/15 20:44
@file: get_clean_detection.py
@description:
"""

import os
import numpy as np
from config import Config

data_root = Config.data_root
for mode in ['train', 'test']:
    video_list = os.listdir(os.path.join(data_root, mode))
    for video_name in video_list:
        print('preocessing {}'.format(video_name))
        det_dir = os.path.join(data_root, mode, video_name, 'cdet')
        if not os.path.exists(det_dir):
            os.makedirs(det_dir)
        clean_detection = list()
        cdet_txt = os.path.join(os.path.join(det_dir, 'det.txt'))

        cdet_file = os.path.join('clean_detections/mot17', mode, video_name, 'det/det.npy')
        cdet_data = np.load(cdet_file, allow_pickle=True).item()
        for frame_index in cdet_data.keys():
            for line in cdet_data[frame_index]:
                clean_detection.append([int(frame_index), -1] + line + [-1, -1, -1])
        clean_detection = np.asarray(clean_detection)
        clean_detection[:, 4] -= clean_detection[:, 2]
        clean_detection[:, 5] -= clean_detection[:, 3]
        np.savetxt(cdet_txt, clean_detection, fmt='%.2f', delimiter=',')
