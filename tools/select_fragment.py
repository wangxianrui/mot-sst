'''
@Author: rayenwang
@Date: 2019-08-14 15:58:25
@Description: 
'''

import argparse
import pandas as pd
import numpy as np
import os
from config import EvalConfig as Config


def get_fragment(track_file):
    track_data = np.loadtxt(track_file, dtype=np.float32, delimiter=',')
    track_ids = np.unique(track_data[:, 1]).astype(np.int)
    video_fragment = []
    temp_fragment = []
    for t_id in track_ids:
        track = track_data[track_data[:, 1] == t_id, :]
        fragment = [np.min(track[:, 0]), np.max(track[:, 0])]
        if len(temp_fragment) == 0 or fragment[0] - temp_fragment[-1][1] > Config.max_interval:
            temp_fragment.append(fragment)
        else:
            temp_fragment[-1] = [min(temp_fragment[-1][0], fragment[0]), max(temp_fragment[-1][1], fragment[1])]
    for fragment in temp_fragment:
        if (fragment[1] - fragment[0]) >= Config.min_duration:
            video_fragment.append(fragment)
            # print('\t' + fragment + '\n')

    with open(track_file, 'w') as file:
        for fragment in video_fragment:
            file.write('fragment: {}\n'.format(fragment))
            mask = np.logical_and(track_data[:, 0] >= fragment[0], track_data[:, 0] <= fragment[1])
            fg_data = track_data[mask, :]
            for line in fg_data:
                file.write('{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1\n'.format(
                    line[0], line[1], line[2], line[3], line[4], line[5], line[6]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, help='train or test')
    args = parser.parse_args()
    res_list = os.listdir(os.path.join(Config.result_dir, args.type, 'txt'))
    for track_file in res_list:
        print(track_file)
        track_file = os.path.join(Config.result_dir, args.type, 'txt', track_file)
        get_fragment(track_file)
