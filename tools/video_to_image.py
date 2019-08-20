import cv2
import os
import numpy as np
import shutil

video_list = [name for name in os.listdir('./') if 'mp4' in name]
for video_name in video_list:
    img_dir = video_name.replace('.mp4', '')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    index = 1

    video = cv2.VideoCapture(video_name)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print(video_name, "  ", frame_count)
    while True:
        rval, frame = video.read()
        if not rval:
            break
        img_name = os.path.join(img_dir, '{0:06}.jpg'.format(index))
        index += 1
        cv2.imencode('.jpg', frame)[1].tofile(img_name)
