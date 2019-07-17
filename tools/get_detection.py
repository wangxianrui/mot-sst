# -*- coding: <encoding name> -*-
"""
@authors: rayenwang
@time: 2019/7/15 20:44
@file: get_detection.py
@description:
    get detetion from dataset, and the we can show the detection result with show_track.py script
"""

import os
from config import Config


def main():
    for type in ['train', 'test']:
        result_dir = os.path.join(Config.data_root, 'detection', type)
        video_list = os.listdir(os.path.join(Config.data_root, type))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for video_name in video_list:
            command = 'cp {} {}'.format(os.path.join(Config.data_root, type, video_name, 'cdet/det.txt'),
                                        os.path.join(result_dir, video_name + '.txt'))
            print(command)
            os.system(command)


if __name__ == '__main__':
    main()
