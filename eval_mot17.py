# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""
import os
import argparse
from tqdm import tqdm
import time
import numpy as np
import torch
from config import EvalConfig as Config
from pipline.mot_eval_dataset import MOTEvalDataset
from network.tracker import SSTTracker


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def eval(args):
    if not os.path.exists(os.path.join(Config.result_dir, args.type, 'txt')):
        os.makedirs(os.path.join(Config.result_dir, args.type, 'txt'))

    video_list = os.listdir(os.path.join(Config.data_root, args.type))
    timer = Timer()
    for vname in video_list:
        if Config.detector not in vname:
            continue

        img_dir = os.path.join(Config.data_root, args.type, vname, 'img1')
        det_file = os.path.join(Config.data_root, args.type, vname, 'det/det.txt')
        res_file = os.path.join(Config.result_dir, args.type, 'txt', vname + '.txt')
        result = list()

        print('start processing {}'.format(res_file))
        tracker = SSTTracker()
        dataset = MOTEvalDataset(image_folder=img_dir, detection_file_name=det_file, min_confidence=0.0)
        dataset_iter = iter(dataset)
        for i in tqdm(range(len(dataset))):
            img, det, index, h, w = next(dataset_iter)
            # ## test
            # import cv2
            # if img is not None:
            #     image = img.permute(1, 2, 0).clone().numpy()
            #     for d in det[index].clone().numpy():
            #         x, y, w, h = d * 900
            #         cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 3)
            #     cv2.imshow('win', image)
            #     cv2.waitKey()
            # continue
            # ##
            if det is None:
                tracker.one_frame_pass()
                continue
            if Config.use_cuda:
                img = img.cuda()
                det = det.cuda()
                index = index.cuda()

            timer.tic()
            tracker.update(img, det, index, i)
            timer.toc()

            # save result
            for t in tracker.tracks:
                n = t.nodes[-1]
                if t.age == 1:
                    b = n.get_box(i, tracker.recorder)
                    result.append([i + 1] + [t.id] + [b[0] * w, b[1] * h, b[2] * w, b[3] * h] + [-1, -1, -1, -1])
        np.savetxt(res_file, np.int_(result), fmt='%i', delimiter=',')
        print('finished processing {}'.format(res_file))
    print('total time {}'.format(timer.total_time))
    print('average time {}'.format(timer.average_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='test', help='eval train or test dataset')
    args = parser.parse_args()
    with torch.no_grad():
        eval(args)
