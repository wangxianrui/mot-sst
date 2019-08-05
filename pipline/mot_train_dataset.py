# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""
import os
import os.path
import cv2
import numpy as np
import pandas as pd
import random
import torch
import torch.utils.data as data
from config import TrainConfig as Config
from .augmentations import SSTTrainAugment


class Node:
    def __init__(self, box, frame_id, next_fram_id=-1):
        self.box = box
        self.frame_id = frame_id
        self.next_frame_id = next_fram_id


class Track:
    def __init__(self, id):
        self.nodes = list()
        self.id = id

    def add_node(self, n):
        if len(self.nodes) > 0:
            self.nodes[-1].next_frame_id = n.frame_id
        self.nodes.append(n)

    def get_node_by_index(self, index):
        return self.nodes[index]


class Tracks:
    def __init__(self):
        self.tracks = list()

    def add_node(self, node, id):
        node_add = False
        track_index = 0
        node_index = 0
        for t in self.tracks:
            if t.id == id:
                t.add_node(node)
                node_add = True
                track_index = self.tracks.index(t)
                node_index = t.nodes.index(node)
                break
        if not node_add:
            t = Track(id)
            t.add_node(node)
            self.tracks.append(t)
            track_index = self.tracks.index(t)
            node_index = t.nodes.index(node)

        return track_index, node_index

    def get_track_by_index(self, index):
        return self.tracks[index]


class GTSingleParser:
    '''
    parser one video sequence
    '''

    def __init__(self, folder, min_visibility=Config.min_visibility,
                 min_gap=Config.min_gap_frame, max_gap=Config.max_gap_frame):
        self.min_gap = min_gap
        self.max_gap = max_gap
        # 1. get the gt path and image folder
        gt_file_path = os.path.join(folder, 'gt/gt.txt')
        self.folder = folder

        # 2. read the gt data
        gt_file = pd.read_csv(gt_file_path, header=None)
        gt_file = gt_file[gt_file[6] == 1]
        gt_file = gt_file[gt_file[8] > min_visibility]
        gt_group = gt_file.groupby(0)
        gt_group_keys = gt_group.indices.keys()
        self.max_frame_index = max(gt_group_keys)
        # 3. update tracks
        self.tracks = Tracks()
        self.recorder = {}
        for key in gt_group_keys:
            det = gt_group.get_group(key).values
            ids = np.array(det[:, 1]).astype(int)
            det = np.array(det[:, 2:6])
            det[:, 2:4] += det[:, :2]

            self.recorder[key - 1] = list()
            # 3.1 update tracks
            for id, d in zip(ids, det):
                node = Node(d, key - 1)
                track_index, node_index = self.tracks.add_node(node, id)
                self.recorder[key - 1].append((track_index, node_index))

    def _getimage(self, frame_index):
        image_path = os.path.join(self.folder, 'img1/{0:06}.jpg'.format(frame_index + 1))
        return cv2.imread(image_path)

    def get_item(self, frame_index):
        '''
        get the current_image, current_boxes, next_image, next_boxes, labels from the frame_index
        :param frame_index:
        :return: current_image, current_boxes, next_image, next_boxes, labels
        '''
        if not frame_index in self.recorder:
            return None, None, None, None, None
        # get current_image, current_box, next_image, next_box and labels
        current_image = self._getimage(frame_index)
        current_boxes = list()
        current = self.recorder[frame_index]
        next_frame_indexes = list()
        current_track_indexes = list()
        # 1. get current box
        for track_index, node_index in current:
            t = self.tracks.get_track_by_index(track_index)
            n = t.get_node_by_index(node_index)
            current_boxes.append(n.box)

            current_track_indexes.append(track_index)
            if n.next_frame_id != -1:
                next_frame_indexes.append(n.next_frame_id)

        # 2. decide the next frame
        # (0.5 probability to choose the farest ones, and other probability to choose the frame between them)
        if len(next_frame_indexes) == 0:
            return None, None, None, None, None
        if len(next_frame_indexes) == 1:
            next_frame_index = next_frame_indexes[0]
        else:
            max_next_frame_index = max(next_frame_indexes)
            is_choose_farest = bool(random.getrandbits(1))
            if is_choose_farest:
                next_frame_index = max_next_frame_index
            else:
                next_frame_index = random.choice(next_frame_indexes)
                gap_frame = random.randint(self.min_gap, self.max_gap)
                temp_frame_index = next_frame_index + gap_frame
                choice_gap = list(range(self.min_gap, self.max_gap))
                if self.min_gap != 0:
                    choice_gap.append(0)
                while temp_frame_index not in self.recorder:
                    gap_frame = random.choice(choice_gap)
                    temp_frame_index = next_frame_index + gap_frame
                next_frame_index = temp_frame_index
        # 3. get next image
        next_image = self._getimage(next_frame_index)

        # 4. get next frame boxes
        next = self.recorder[next_frame_index]
        next_boxes = list()
        next_track_indexes = list()
        for track_index, node_index in next:
            t = self.tracks.get_track_by_index(track_index)
            next_track_indexes.append(track_index)
            n = t.get_node_by_index(node_index)
            next_boxes.append(n.box)

        # 5. get the labels
        current_track_indexes = np.array(current_track_indexes)
        next_track_indexes = np.array(next_track_indexes)
        labels = np.repeat(np.expand_dims(np.array(current_track_indexes), axis=1), len(next_track_indexes), axis=1) \
                 == np.repeat(np.expand_dims(np.array(next_track_indexes), axis=0), len(current_track_indexes), axis=0)

        # 6. return all values
        # 6.1 change boxes format
        current_boxes = np.array(current_boxes)
        next_boxes = np.array(next_boxes)
        # 6.2 return the corresponding values
        return current_image, current_boxes, next_image, next_boxes, labels

    def __len__(self):
        return self.max_frame_index


class GTParser:
    '''
    parser all video sequence
    '''

    def __init__(self):
        # 1. get all the folders
        data_root = os.path.join(Config.data_root, 'train')
        all_folders = sorted([os.path.join(data_root, i)
                              for i in os.listdir(data_root)
                              if os.path.isdir(os.path.join(data_root, i)) and i.find(Config.detector) != -1])
        # 2. parser video sequence
        self.parsers = [GTSingleParser(folder) for folder in all_folders]

        # 3. get some basic information
        self.lens = [len(p) for p in self.parsers]
        self.len = sum(self.lens)

    def __len__(self):
        # get the length of all the matching frame
        return self.len

    def __getitem__(self, item):
        if item < 0:
            return None, None, None, None, None
        # 1. find the parser
        total_len = 0
        index = 0
        current_item = item
        for l in self.lens:
            total_len += l
            if item < total_len:
                break
            else:
                index += 1
                current_item -= l

        # 2. get items
        if index >= len(self.parsers):
            return None, None, None, None, None
        return self.parsers[index].get_item(current_item)


class MOTTrainDataset(data.Dataset):
    '''
    The class is the dataset for train, which read gt.txt file and rearrange them as the tracks set.
    it can be selected from the specified frame
    '''

    def __init__(self):
        self.transform = SSTTrainAugment(Config.sst_dim, Config.mean_pixel)
        self.parser = GTParser()

    def __getitem__(self, item):
        current_image, current_box, next_image, next_box, labels = self.parser[item]

        while labels is None:
            current_image, current_box, next_image, next_box, labels = \
                self.parser[random.randint(0, len(self.parser))]

        # change the label to max_object x max_object
        labels = np.pad(labels, [(0, Config.max_object - labels.shape[0]),
                                 (0, Config.max_object - labels.shape[1])], mode='constant', constant_values=0)
        return self.transform(current_image, next_image, current_box, next_box, labels)

    def __len__(self):
        return len(self.parser)


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
