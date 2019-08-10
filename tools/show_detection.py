# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: 2019/8/8 16:50
@file: show_detection.py
@description:
"""
import os
import numpy as np
import cv2

confidence_thershold = 0.3
video_dir = 'D:/movies_08_06/test/SG教会我爱你第一季-01'

detection = np.loadtxt(os.path.join(video_dir, 'det', 'det.txt'), delimiter=',')
detection = detection[detection[:, 6] > confidence_thershold, :]
frame_index = np.unique(detection[:, 0]).astype(np.int)
cv2.namedWindow('win', cv2.WINDOW_NORMAL)
for index in frame_index:
    img = cv2.imdecode(np.fromfile(os.path.join(video_dir, 'img1', '{0:06}.jpg'.format(index)), dtype=np.uint8), -1)
    frame_det = detection[detection[:, 0] == index, :]
    print(index, '\t', len(frame_det))
    for det in frame_det:
        x, y, w, h = [int(n) for n in det[2:6]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(img, str(det[6]), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow('win', img)
    cv2.waitKey()
