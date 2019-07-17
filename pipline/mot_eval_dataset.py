import os
import pandas as pd
import cv2
import torch.utils.data
from config import Config


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
        return len(self.detection_group_keys)

    def get_detection_by_index(self, index):
        if index > len(self.detection_group_keys) or self.detection_group_keys.count(index) == 0:
            return None
        return self.detection_group.get_group(index).values

    def get_image_by_index(self, index):
        if index > len(self.detection_group_keys):
            return None
        return cv2.imread(self.image_format.format(index))

    def __getitem__(self, item):
        image = self.get_image_by_index(item + 1)
        detection = self.get_detection_by_index(item + 1)
        return self.transform(image, detection)

    @staticmethod
    def transform(image, detection):
        if len(detection) > Config.max_object:
            detection = detection[:Config.max_object, :]
        h, w, _ = image.shape
        detection[:, [2, 4]] /= float(w)
        detection[:, [3, 5]] /= float(h)
        detection = detection[:, 2:6]
        return image, detection
