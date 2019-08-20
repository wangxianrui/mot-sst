'''
@Author: rayenwang
@Date: 2019-08-12 21:20:01
@Description: 
    format.py
    imgdir1
    imgdir2
    ...
    detectiontxt
'''

import os
import shutil

detection_txt = 'xianrui_images_test.log'

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
    print('remove det in {}'.format(video_name))
    det_dir = os.path.join(video_name, 'det')
    if os.path.exists(det_dir):
        shutil.rmtree(det_dir)
    os.makedirs(det_dir)

# create detection file
with open(detection_txt, encoding='utf-8') as det_file:
    file_data = det_file.read().splitlines()

    video_name = None
    for line in file_data:
        line = line.split()
        temp_name = line[0].split('/')[-2]
        if temp_name != video_name:
            print(temp_name)
            video_name = temp_name
            file_name = os.path.join(video_name, 'det', 'det.txt')
            file = open(file_name, 'a')
        frame_index = int(os.path.split(line[0])[-1][:-4])
        # confidence, x, y, x, y
        det = [float(n) for n in line[2:]] + [float(line[1])]
        string = '{},-1,{},{},{},{},{:.4f},{},-1,-1\n'.format(frame_index, det[1], det[2], det[3] - det[1],
                                                              det[4] - det[2], det[0], det[5])
        file.write(string)
'''
/data2/mionyu/movies_08_06/SG教会我爱你第一季-10/000070.jpg 3 0.0553201 34 391 50 402
'''
