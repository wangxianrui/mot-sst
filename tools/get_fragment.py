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
    track_group = pd.read_csv(track_file, header=None).groupby(1)
    track_ids = track_group.indices.keys()
    video_fragment = []
    for id in track_ids:
        track = track_group.get_group(id).values
        fragment = [np.min(track[:, 0]), np.max(track[:, 0])]
        if len(video_fragment) == 0 or fragment[0] - video_fragment[-1][1] > Config.max_interval:
            video_fragment.append(fragment)
        else:
            video_fragment[-1] = [min(video_fragment[-1][0], fragment[0]), max(video_fragment[-1][1], fragment[1])]

    for fragment in video_fragment:
        if (fragment[1] - fragment[0]) > Config.min_duration:
            print('\t' + str(fragment))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, help='train or test')
    args = parser.parse_args()
    res_list = os.listdir(os.path.join(Config.result_dir, args.type, 'txt'))
    for track_file in res_list:
        print(track_file)
        track_file = os.path.join(Config.result_dir, args.type, 'txt', track_file)
        get_fragment(track_file)
