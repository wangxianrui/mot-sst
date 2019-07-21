# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: 2019/7/19 11:12
@file: get_max_object.py
@description:
"""

import os
import pandas as pd
from config import TrainConfig as Config


def main():
    data_root = os.path.join(Config.data_root, 'train')
    video_list = os.listdir(data_root)
    max_object = 0
    for video_name in video_list:
        if Config.detector not in video_name:
            continue
        gt_file_path = os.path.join(data_root, video_name, 'gt/gt.txt')
        gt_file = pd.read_csv(gt_file_path, header=None)
        gt_file = gt_file[gt_file[6] == 1]
        gt_file = gt_file[gt_file[8] > Config.min_visibility]
        gt_group = gt_file.groupby(0)
        gt_group_keys = gt_group.indices.keys()
        frame_len = [len(gt_group.indices[key]) for key in gt_group_keys]
        max_len = max(frame_len)
        print('max_object in {}: {}'.format(video_name, max_len))
        if max_len > max_object:
            max_object = max_len
    print('max_object in all video: {}'.format(max_object))


if __name__ == '__main__':
    main()
