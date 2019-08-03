# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""
import os
import pandas as pd
import cv2
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from config import EvalConfig as Config


class MOTEvalDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, detection_file_name, min_confidence=None):
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.image_format = os.path.join(self.image_folder, '{0:06d}.jpg')
        self.detection = pd.read_csv(self.detection_file_name, header=None)
        if min_confidence is not None:
            self.detection = self.detection[self.detection[6] > min_confidence]
        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())

    def __len__(self):
        return len(os.listdir(self.image_folder))

    def get_detection_by_index(self, index):
        if self.detection_group_keys.count(index) == 0:
            return None
        return self.detection_group.get_group(index).values

    def get_image_by_index(self, index):
        return cv2.imread(self.image_format.format(index)).astype(np.float32)

    def __getitem__(self, item):
        image = self.get_image_by_index(item + 1)
        detection = self.get_detection_by_index(item + 1)
        if detection is None:
            return None, None, None, None, None
        return self.transform(image, detection)

    @staticmethod
    def transform(image, detection):
        # image
        h, w, _ = image.shape
        image = cv2.resize(image, Config.image_size)
        image -= Config.mean_pixel
        image /= 127.5
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)

        # detection
        detection = torch.from_numpy(detection).float()
        det_num = detection.shape[0]
        detection[:, [2, 4]] /= float(w)
        detection[:, [3, 5]] /= float(h)
        if det_num > Config.max_object:
            index = torch.argsort(detection, descending=False, dim=0)[:, 6]
            detection = detection[index[:Config.max_object], :]
        else:
            detection = F.pad(detection, [0, 0, 0, Config.max_object - detection.shape[0]], value=-1)
        detection = detection[:, 2:6]

        # shuffle
        index = np.arange(Config.max_object)
        np.random.shuffle(index)
        detection = detection[index, :]
        valid_mask = torch.from_numpy(index < det_num).float()
        valid_index = torch.nonzero(valid_mask).squeeze(dim=1)
        return image, detection, valid_index, h, w
