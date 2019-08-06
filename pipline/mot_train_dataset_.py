# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: 2019/8/5 11:35
@file: mot_train_dataset_.py
@description:
"""
import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.utils.data as data

from config import TrainConfig as Config
from .augmentations import SSTTrainAugment


class MOTTrainDataset(data.Dataset):
    def __init__(self):
        super(MOTTrainDataset, self).__init__()
        self.transform = SSTTrainAugment(Config.sst_dim, Config.mean_pixel)
        self.video_list = [name for name in os.listdir(os.path.join(Config.data_root, 'train')) if Config.detector in name]
        self.video_len = [len(os.listdir(os.path.join(Config.data_root, 'train', name, 'img1')))
                          for name in self.video_list]

    def __getitem__(self, item):
        # video index and frame index
        video_index = 0
        while item >= self.video_len[video_index]:
            item -= self.video_len[video_index]
            video_index += 1
        current_frame_index = item + 1
        next_frame_index = item + random.randint(-Config.max_gap_frame, Config.max_gap_frame)
        if next_frame_index <= 0:
            next_frame_index = 1
        if next_frame_index > self.video_len[video_index]:
            next_frame_index = self.video_len[video_index]

        # load gt file
        gt_file = os.path.join(Config.data_root, 'train', self.video_list[video_index], 'gt/gt.txt')
        gt_parser = pd.read_csv(gt_file, header=None)
        gt_parser = gt_parser[gt_parser[6] == 1]
        gt_parser = gt_parser[gt_parser[8] > Config.min_visibility]
        gt_group = gt_parser.groupby(0)
        gt_group_keys = gt_group.indices.keys()
        if current_frame_index == next_frame_index or current_frame_index not in gt_group_keys \
                or next_frame_index not in gt_group_keys:
            return self.__getitem__(random.randint(0, self.__len__()))

        # load image
        image_path_format = os.path.join(Config.data_root, 'train', self.video_list[video_index], 'img1', '{0:06}.jpg')
        current_image = cv2.imread(image_path_format.format(current_frame_index))
        next_image = cv2.imread(image_path_format.format(next_frame_index))

        # load detection and track_id
        current_detection = gt_group.get_group(current_frame_index).values
        current_id = np.asarray(current_detection)[:, 1].astype(np.int).reshape(-1, 1)
        current_detection = np.asarray(current_detection)[:, 2:6]
        current_detection[:, 2:4] += current_detection[:, :2]
        next_detection = gt_group.get_group(next_frame_index).values
        next_id = np.asarray(next_detection)[:, 1].astype(np.int).reshape(1, -1)
        next_detection = np.asarray(next_detection)[:, 2:6]
        next_detection[:, 2:4] += next_detection[:, :2]

        # create labels
        labels = np.asarray(current_id == next_id, dtype=np.int)
        labels = np.pad(labels, [(0, Config.max_object - labels.shape[0]),
                                 (0, Config.max_object - labels.shape[1])], mode='constant', constant_values=0)
        return self.transform(current_image, next_image, current_detection, next_detection, labels)

    def __len__(self):
        return sum(self.video_len)


def collate_fn(batch):
    img_pre = []
    img_next = []
    boxes_pre = []
    boxes_next = []
    labels = []
    indexes_pre = []
    indexes_next = []
    for sample in batch:
        img_pre.append(sample[0])
        img_next.append(sample[1])
        boxes_pre.append(sample[2][0].float())
        boxes_next.append(sample[3][0].float())
        labels.append(sample[4].float())
        indexes_pre.append(sample[2][1].float())
        indexes_next.append(sample[3][1].float())
    return torch.stack(img_pre, 0), torch.stack(img_next, 0), torch.stack(boxes_pre, 0), torch.stack(boxes_next, 0), \
           torch.stack(labels, 0), torch.stack(indexes_pre, 0).unsqueeze(1), torch.stack(indexes_next, 0).unsqueeze(1)
