# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""
import random
import argparse
import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from config import Config


def main(args):
    data_root = os.path.join(Config.data_root, args.type)
    colorList = get_spaced_colors(100)
    random.shuffle(colorList)
    video_list = os.listdir(data_root)

    for video_name in video_list:
        txt_file = os.path.join(data_root, video_name, 'gt/gt.txt')
        avi_file = os.path.join(args.save_dir, args.type, video_name + '.avi')
        img_dir = os.path.join(data_root, video_name, 'img1')
        temp_img = cv2.imread(os.path.join(img_dir, '000001.jpg'))
        h, w, _ = temp_img.shape
        vwriter = cv2.VideoWriter(avi_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (w, h))

        res_raw = pd.read_csv(txt_file, sep=',', header=None)
        res_raw = np.array(res_raw).astype(np.float32)
        res_raw = res_raw[res_raw[:, -3] == 1, :]
        res_raw = res_raw[res_raw[:, -1] >= 0.3, :]
        res_raw[:, 0:6] = np.array(res_raw[:, 0:6]).astype(np.int)
        n_frame = max(res_raw[:, 0])
        print('video_name: {} || total number of frames: {}'.format(video_name, n_frame))
        for t in tqdm(range(1, int(n_frame))):
            img_name = os.path.join(img_dir, str(t).zfill(6) + '.jpg')
            img = cv2.imread(img_name)
            overlay = img.copy()
            row_ind = np.where(res_raw[:, 0] == t)[0]
            for i in range(0, row_ind.shape[0]):
                id = int(max(res_raw[row_ind[i], 1], 0))
                color_ind = id % len(colorList)

                # plot the line
                row_ind_line = np.where((res_raw[:, 0] > t - 50) & (res_raw[:, 0] < t + 1) & (res_raw[:, 1] == id))[0]

                # plot the rectangle
                for j in range(0, row_ind_line.shape[0], 5):
                    line_xc = int(res_raw[row_ind_line[j], 2] + 0.5 * res_raw[row_ind_line[j], 4])
                    line_yc = int(res_raw[row_ind_line[j], 3] + res_raw[row_ind_line[j], 5])
                    bb_w = 5
                    line_x1 = line_xc - bb_w
                    line_y1 = line_yc - bb_w
                    line_x2 = line_xc + bb_w
                    line_y2 = line_yc + bb_w
                    cv2.rectangle(overlay, (line_x1, line_y1), (line_x2, line_y2), colorList[color_ind], -1)

                    t_past = res_raw[row_ind_line[j], 0]
                    alpha = 1 - (t - t_past) / 80  # Transparency factor.
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                    overlay = img.copy()

            for i in range(0, row_ind.shape[0]):
                id = int(res_raw[row_ind[i], 1])
                bb_x1 = int(res_raw[row_ind[i], 2])
                bb_y1 = int(res_raw[row_ind[i], 3])
                bb_x2 = int(res_raw[row_ind[i], 2] + res_raw[row_ind[i], 4])
                bb_y2 = int(res_raw[row_ind[i], 3] + res_raw[row_ind[i], 5])
                color_ind = id % len(colorList)
                cv2.rectangle(overlay, (bb_x1, bb_y1), (bb_x2, bb_y2), colorList[color_ind], 3)
                cv2.putText(overlay, str(id), (bb_x1, bb_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, colorList[color_ind], 3)
            vwriter.write(overlay)


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Images from Results')
    parser.add_argument('--save_dir', default='ground_truth')
    parser.add_argument('--type', required=True)
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.save_dir, args.type)):
        os.makedirs(os.path.join(args.save_dir, args.type))

    main(args)
