'''
@Author: rayenwang
@Date: 2019-08-02 14:28:06
@Description: 
'''

import os
import shutil
from config import Config

gt_dir = Config.data_root
gt_suffix = '.rec'
img_suffix = '.jpg'

video_list = os.listdir(gt_dir)
for video_name in video_list:
    video_dir = os.path.join(gt_dir, video_name)
    print(video_dir)
    img_list = [name for name in os.listdir(video_dir) if img_suffix in name]
    rec_list = [name for name in os.listdir(video_dir) if gt_suffix in name]

    # move img
    img_dir = os.path.join(video_dir, 'img1')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for img_name in img_list:
        shutil.move(os.path.join(video_dir, img_name), os.path.join(img_dir, img_name))

    # move rec file
    rec_dir = os.path.join(video_dir, 'gt')
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)
    for rec_name in rec_list:
        shutil.move(os.path.join(video_dir, rec_name), os.path.join(rec_dir, rec_name))

    # create gt file
    with open(os.path.join(rec_dir, 'gt.txt'), 'w') as file:
        for rec_name in rec_list:
            frame_index = int(rec_name.replace(gt_suffix, ''))
            frame_info = []
            with open(os.path.join(rec_dir, rec_name)) as in_file:
                data = in_file.read().splitlines()
                for line in data:
                    line = [int(n) for n in line.split()]
                    line[3] -= line[1]
                    line[4] -= line[2]
                    frame_info.append(str([frame_index] + line[:-1] + [1, 1, 1])[1:-1] + '\n')
            file.writelines(frame_info)

    # create det file
    det_dir = os.path.join(video_dir, 'det')
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)
    shutil.copy(os.path.join(rec_dir, 'gt.txt'), os.path.join(det_dir, 'det.txt'))
