# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: 2019/8/7 10:23
@file: get_fragment.py
@description:
"""
import argparse
import pandas as pd
import numpy as np
import os
from config import EvalConfig as Config


def get_fragment(track_file, file_name):
    track_group = pd.read_csv(track_file, header=None).groupby(1)
    track_ids = track_group.indices.keys()
    video_fragment = []
    for id in track_ids:
        track = track_group.get_group(id).values
        fragment = [np.min(track[:, 0]), np.max(track[:, 0])]
        if len(video_fragment) == 0 or fragment[0] - video_fragment[-1][1] > Config.max_interval:
            video_fragment.append(fragment)
        else:
            last_fragment = video_fragment[-1]
            video_fragment[-1] = [min(last_fragment[0], fragment[0]), max(last_fragment[1], fragment[1])]
    with open(file_name, 'a') as file:
        file.write(track_file[:-4] + '\n')
        for fragment in video_fragment:
            if (fragment[1] - fragment[0]) > Config.min_duration:
                file.write('\t' + str(fragment) + '\n')
        file.write('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, help='train or test')
    args = parser.parse_args()
    file_name = os.path.join(Config.result_dir, 'interval.txt')
    if os.path.exists(file_name):
        os.remove(file_name)
    res_list = os.listdir(os.path.join(Config.result_dir, args.type, 'txt'))
    for track_file in res_list:
        track_file = os.path.join(Config.result_dir, args.type, 'txt', track_file)
        get_fragment(track_file, file_name)
