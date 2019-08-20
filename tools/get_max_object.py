'''
@Author: rayenwang
@Date: 2019-08-08 16:02:57
@Description: 
'''

import os
import pandas as pd
import argparse
from config import EvalConfig as Config


def main(args):
    data_root = os.path.join(Config.data_root, args.type)
    video_list = os.listdir(data_root)
    max_object = 0
    for video_name in video_list:
        if Config.detector not in video_name:
            continue
        gt_file_path = os.path.join(data_root, video_name, 'det/det.txt')
        gt_file = pd.read_csv(gt_file_path, header=None)
        gt_file = gt_file[gt_file[6] > Config.low_confidence]
        gt_group = gt_file.groupby(0)
        gt_group_keys = gt_group.indices.keys()
        frame_len = [len(gt_group.indices[key]) for key in gt_group_keys]
        max_len = max(frame_len)
        print('max_object in {}: {}'.format(video_name, max_len))
        if max_len > max_object:
            max_object = max_len
    print('max_object in all video: {}'.format(max_object))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, help='train or test')
    args = parser.parse_args()
    main(args)
