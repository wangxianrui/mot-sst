# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""
import torch
import math
from scipy.optimize import linear_sum_assignment
from config import EvalConfig as Config
from .sst import build_sst


class FeatureRecorder:
    '''
    @Description:
        record all informations with frame_index
        include features, bboxes, similarity and iou ...
    @Parameter:
        boxes[frame_index][det_index]
        fetures[frame_index][det_index]
        iou[frame_index[preframe]
        similarity[frame_index][preframe]
    @Return:
    '''

    def __init__(self):
        self.all_frame_index = []
        self.all_features = {}
        self.all_boxes = {}
        self.all_mask = {}
        self.all_similarity = {}
        self.all_iou = {}

    def update(self, sst, frame_index, features, boxes, valid_mask):
        if len(self.all_frame_index) == Config.max_track_frame:
            del_frame = self.all_frame_index[0]
            del self.all_features[del_frame]
            del self.all_boxes[del_frame]
            del self.all_mask[del_frame]
            del self.all_similarity[del_frame]
            del self.all_iou[del_frame]
            del self.all_frame_index[0]
        # add new attribute
        self.all_frame_index.append(frame_index)
        self.all_features[frame_index] = features
        self.all_boxes[frame_index] = boxes
        self.all_mask[frame_index] = valid_mask

        self.all_similarity[frame_index] = {}
        for pre_index in self.all_frame_index[:-1]:
            pre_similarity = sst.get_similarity(self.all_features[pre_index].clone(), self.all_mask[pre_index].clone(),
                                                features.clone(), valid_mask.clone())
            self.all_similarity[frame_index][pre_index] = pre_similarity

        self.all_iou[frame_index] = {}
        for pre_index in self.all_frame_index[:-1]:
            iou = self.get_iou(self.all_boxes[pre_index].clone(), boxes.clone())
            self.all_iou[frame_index][pre_index] = iou

    def get_iou(self, bboxes1, bboxes2):
        bboxes1[:, 2:] += bboxes1[:, :2]
        bboxes2[:, 2:] += bboxes2[:, :2]
        bboxes1.unsqueeze_(dim=1)
        bboxes2.unsqueeze_(dim=0)
        bboxes1 = bboxes1.permute(2, 0, 1)
        bboxes2 = bboxes2.permute(2, 0, 1)
        # Intersection bbox and volume.
        int_xmin = torch.max(bboxes1[0], bboxes2[0])
        int_ymin = torch.max(bboxes1[1], bboxes2[1])
        int_xmax = torch.min(bboxes1[2], bboxes2[2])
        int_ymax = torch.min(bboxes1[3], bboxes2[3])

        int_h = torch.clamp_min(int_ymax - int_ymin, 0)
        int_w = torch.clamp_min(int_xmax - int_xmin, 0)
        int_vol = int_h * int_w
        # Union volume.
        vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
        vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
        iou = int_vol / (vol1 + vol2 - int_vol)
        return iou


class Node:
    def __init__(self, frame_index, det_index):
        self.frame_index = frame_index
        self.det_index = det_index

    def get_box(self, frame_index, recoder):
        if frame_index - self.frame_index >= Config.max_track_frame:
            return None
        return recoder.all_boxes[self.frame_index][self.det_index, :]

    def get_iou(self, frame_index, recoder, box_id):
        if frame_index - self.frame_index >= Config.max_track_frame:
            return None
        return recoder.all_iou[frame_index][self.frame_index][self.det_index, box_id]


class Track:
    def __init__(self, track_id):
        self.nodes = list()
        self.id = track_id
        self.age = 0

    def add_node(self, frame_index, recorder, node):
        # if len(self.nodes) > 0:
        #     n = self.nodes[-1]
        #     iou = n.get_iou(frame_index, recorder, node.det_index)
        #     delta_frame = frame_index - n.frame_index
        #     # filter out with low iou
        #     if iou < 0.5 ** delta_frame:
        #         return
        self.nodes.append(node)
        self.age = 0

    def get_similarity(self, frame_index, recorder, valid_index):
        similarity = []
        for n in self.nodes:
            f = n.frame_index
            id = n.det_index
            if frame_index - f >= Config.max_track_frame:
                continue
            feature_index = torch.cat([valid_index, torch.tensor([-1]).to(valid_index.device)])
            feture_similarity = recorder.all_similarity[frame_index][f][id, :][feature_index]
            similarity += [feture_similarity]
        similarity = torch.stack(similarity, dim=0)
        similarity = torch.mean(similarity, dim=0)
        # iou filter
        f = self.nodes[-1].frame_index
        id = self.nodes[-1].det_index
        iou_similarity = recorder.all_iou[frame_index][f][id, :][valid_index]
        similarity[:-1][iou_similarity < Config.iou_threshold ** (frame_index - f)] = 0
        return similarity


class SSTTracker:
    def __init__(self):
        self.sst = build_sst()
        self.id_pool = 0
        self.tracks = list()
        self.recorder = FeatureRecorder()
        self.load_model()

    def load_model(self):
        print('loading model from {}'.format(Config.model_path))
        if Config.use_cuda:
            self.sst.load_state_dict(torch.load(Config.model_path)['state_dict'])
            self.sst = self.sst.cuda()
        else:
            self.sst.load_state_dict(torch.load(Config.model_path, map_location='cpu')['state_dict'])
        self.sst.eval()

    def add_track(self, track):
        self.tracks.append(track)

    def get_similarity(self, frame_index, recorder, valid_index):
        similarity = []
        for t in self.tracks:
            s = t.get_similarity(frame_index, recorder, valid_index)
            similarity += [s]
        return torch.stack(similarity, dim=0)

    def one_frame_pass(self):
        # remove old track
        temp_tracks = self.tracks.copy()
        self.tracks.clear()
        for track in temp_tracks:
            track.age += 1
            if track.age < Config.max_track_frame:
                self.add_track(track)

    def update(self, image, detection, valid_mask, frame_index):
        '''
        @Description: 
        @Parameter: 
            image: 3 * dim * dim
            detection: maxN * 5, x,y,w,h,confidence
            valid_index: maxN + 1
        @Return: 
        '''
        valid_index = torch.nonzero(valid_mask).squeeze(dim=1)
        confidence = detection[:, -1]
        detection = detection[:, :-1]
        features = self.sst.forward_feature(image.unsqueeze(0), self.convert_detection(detection.clone()))
        self.recorder.update(self.sst, frame_index, features, detection, valid_mask)
        track_num = len(self.tracks)
        det_num = valid_index.shape[0]

        # first frame
        if frame_index == 0 or track_num == 0:
            for index in valid_index:
                # add new track
                if confidence[index] > Config.high_confidence:
                    track = Track(self.id_pool)
                    self.id_pool += 1
                    node = Node(frame_index, index)
                    track.add_node(frame_index, self.recorder, node)
                    self.add_track(track)
        else:
            # similarity between track and detection    track_num * det_num + 1
            similarity = self.get_similarity(frame_index, self.recorder, valid_index)
            additional = torch.repeat_interleave(similarity[:, -1].reshape(track_num, 1), track_num - 1, dim=1)
            similarity = torch.cat([similarity, additional], dim=1).detach().cpu().numpy()

            # linear_sum_assignment
            row_index, col_index = linear_sum_assignment(-similarity)
            col_index[col_index >= det_num] = -1
            # update tracks
            for i in range(track_num):
                track = self.tracks[i]
                if col_index[i] != -1:
                    det_index = valid_index[col_index[i]]
                    node = Node(frame_index, det_index)
                    track.add_node(frame_index, self.recorder, node)
            # add new tracks
            for j in range(det_num):
                if j not in col_index:
                    index = valid_index[j]
                    # add new track
                    if confidence[index] > Config.high_confidence:
                        track = Track(self.id_pool)
                        self.id_pool += 1
                        node = Node(frame_index, index)
                        track.add_node(frame_index, self.recorder, node)
                        self.add_track(track)
        self.one_frame_pass()

    def convert_detection(self, detection):
        # convert detection to -1 -- 1
        center = (2 * detection[:, 0:2] + detection[:, 2:4]) - 1.0
        center.unsqueeze_(0)
        center.unsqueeze_(2)
        center.unsqueeze_(3)
        return center
