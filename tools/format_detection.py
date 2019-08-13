# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: 2019/8/7 15:46
@file: format_detection.py
@description:
    format.py
    imgdir1
    imgdir2
    ...
    detectiontxt
"""
import os
import shutil

detection_txt = 'xianrui_images_test_0807.log'

# move img
video_list = [name for name in os.listdir('.') if os.path.isdir(name)]
for video_name in video_list:
    print('moving imgs in {}'.format(video_name))
    img_dir = os.path.join(video_name, 'img1')
    img_list = [name for name in os.listdir(video_name) if 'jpg' in name]
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    for img in img_list:
        shutil.move(os.path.join(video_name, img), os.path.join(img_dir, img))

# create detection file
with open(detection_txt, encoding='utf-8') as det_file:
    file_data = det_file.read().splitlines()

    video_name = None
    for line in file_data:
        line = line.split()
        if int(line[1]) != 3:
            continue
        temp_name = line[0].split('/')[-2]
        if temp_name != video_name:
            print(temp_name)
            video_name = temp_name
            if not os.path.exists(os.path.join(video_name, 'det')):
                os.makedirs(os.path.join(video_name, 'det'))
            file_name = os.path.join(video_name, 'det', 'det.txt')
            if os.path.exists(file_name):
                os.remove(file_name)
            file = open(file_name, 'a')
        frame_index = int(os.path.split(line[0])[-1][:-4])
        # confidence, x, y, x, y
        det = [float(n) for n in line[2:]]
        string = '{},-1,{},{},{},{},{:.4f},-1,-1,-1\n'.format(frame_index, det[1], det[2], det[3] - det[1], det[4] - det[2], det[0])
        file.write(string)
'''
/data2/mionyu/movies_08_06/SG教会我爱你第一季-10/000070.jpg 3 0.0553201 34 391 50 402
'''
