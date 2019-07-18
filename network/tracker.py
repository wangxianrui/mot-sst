'''
@Author: rayenwang
@Date: 2019-07-17 14:58:49
@LastEditTime: 2019-07-18 11:15:49
@Description: 
'''
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import EvalConfig as Config
from .sst import build_sst


def get_iou(pre_boxes, next_boxes):
    h = len(pre_boxes)
    w = len(next_boxes)
    if h == 0 or w == 0:
        return []
    iou = np.zeros((h, w), dtype=float)
    for i in range(h):
        b1 = np.copy(pre_boxes[i, :])
        b1[2:] = b1[:2] + b1[2:]
        for j in range(w):
            b2 = np.copy(next_boxes[j, :])
            b2[2:] = b2[:2] + b2[2:]
            delta_h = min(b1[2], b2[2]) - max(b1[0], b2[0])
            delta_w = min(b1[3], b2[3]) - max(b1[1], b2[1])
            if delta_h < 0 or delta_w < 0:
                expand_area = (max(b1[2], b2[2]) - min(b1[0], b2[0])) * (max(b1[3], b2[3]) - min(b1[1], b2[1]))
                area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou[i, j] = -(expand_area - area) / area
            else:
                overlap = delta_h * delta_w
                area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - max(overlap, 0)
                iou[i, j] = overlap / area
    return iou


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
        self.all_frame_index = np.array([], dtype=int)
        self.all_features = {}
        self.all_boxes = {}
        self.all_similarity = {}
        self.all_iou = {}

    def update(self, sst, frame_index, features, boxes):
        if len(self.all_frame_index) == Config.max_track_frame:
            del_frame = self.all_frame_index[0]
            del self.all_features[del_frame]
            del self.all_boxes[del_frame]
            del self.all_similarity[del_frame]
            del self.all_iou[del_frame]
            self.all_frame_index = self.all_frame_index[1:]
        # add new attribute
        self.all_frame_index = np.append(self.all_frame_index, frame_index)
        self.all_features[frame_index] = features
        self.all_boxes[frame_index] = boxes

        self.all_similarity[frame_index] = {}
        for pre_index in self.all_frame_index[:-1]:
            pre_similarity = sst.get_similarity(self.all_features[pre_index], features)
            self.all_similarity[frame_index][pre_index] = pre_similarity

        self.all_iou[frame_index] = {}
        for pre_index in self.all_frame_index[:-1]:
            iou = get_iou(self.all_boxes[pre_index], boxes)
            self.all_iou[frame_index][pre_index] = iou


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
    _id_pool = 0

    def __init__(self):
        self.nodes = list()
        self.id = Track._id_pool
        self.age = 0
        Track._id_pool += 1

    def add_node(self, frame_index, recorder, node):
        if len(self.nodes) > 0:
            n = self.nodes[-1]
            iou = n.get_iou(frame_index, recorder, node.det_index)
            delta_frame = frame_index - n.frame_index
            if iou < 0.5**delta_frame:
                return
        self.nodes.append(node)
        self.age = 0

    def get_similarity(self, frame_index, recorder):
        if len(self.nodes) == 0:
            return None
        similarity = []
        for n in self.nodes:
            f = n.frame_index
            id = n.det_index
            if frame_index - f >= Config.max_track_frame:
                continue
            similarity += [recorder.all_similarity[frame_index][f][id, :]]
        return np.sum(np.array(similarity), axis=0)


class AllTrack:
    def __init__(self):
        self.tracks = list()

    def get_item_by_trackid(self, trackid):
        for t in self.tracks:
            if t.id == trackid:
                return t
        return None

    def get_similarity(self, frame_index, recorder):
        # track_ids = []
        similarity = []
        for t in self.tracks:
            s = t.get_similarity(frame_index, recorder)
            similarity += [s]
            # track_ids += [t.id]
        return np.array(similarity)  # , np.array(track_ids)

    def one_frame_pass(self):
        keep_track_set = list()
        for i, t in enumerate(self.tracks):
            t.age += 1
            if t.age > Config.max_track_frame:
                continue
            keep_track_set.append(i)
        self.tracks = [self.tracks[i] for i in keep_track_set]

    def add_track(self, track):
        self.tracks.append(track)
        # TODO
        # volatile


class SSTTracker:
    def __init__(self):
        self.sst = build_sst()
        self.all_track = AllTrack()
        self.recorder = FeatureRecorder()
        # self.load_model()

    def load_model(self):
        if Config.use_cuda:
            self.sst.load_state_dict(torch.load(Config.model_path)['state_dict'])
            self.sst = self.sst.cuda()
        else:
            self.sst.load_state_dict(torch.load(Config.model_path, map_location='cpu')['state_dict'])
        self.sst.eval()

    def convert_detection(self, detection):
        # convert detection to -1 -- 1
        center = (2 * detection[:, 0:2] + detection[:, 2:4]) - 1.0
        center = torch.from_numpy(center.astype(np.float32)).float()
        center.unsqueeze_(0)
        center.unsqueeze_(2)
        center.unsqueeze_(3)
        return center

    def update(self, image, detection, frame_index):
        features = self.sst.forward_feature(image, self.convert_detection(np.copy(detection)))
        self.recorder.update(self.sst, frame_index, features, detection)
        track_num = len(self.all_track.tracks)
        det_num = detection.shape[0]

        # first frame
        if frame_index == 0 or track_num == 0:
            for i in range(det_num):
                track = Track()
                node = Node(frame_index, i)
                track.add_node(frame_index, self.recorder, node)
                self.all_track.add_track(track)
        else:
            # similarity between track and detection    track_num * det_num
            similarity = self.all_track.get_similarity(frame_index, self.recorder)
            additional = np.repeat(np.reshape(np.min(similarity, axis=1), [track_num, 1]), track_num - 1, axis=1)
            similarity = np.concatenate([similarity, additional], axis=1)
            # linear_sum_assignment
            row_index, col_index = linear_sum_assignment(-similarity)
            col_index[col_index >= det_num] = -1
            # update tracks
            for i in range(track_num):
                track = self.all_track.tracks[i]
                det_index = col_index[i]
                if det_index != -1:
                    node = Node(frame_index, det_index)
                    track.add_node(frame_index, self.recorder, node)
            # add new tracks
            for j in range(det_num):
                if j not in col_index:
                    node = Node(frame_index, j)
                    track = Track()
                    track.add_node(frame_index, self.recorder, node)
                    self.all_track.add_track(track)
        self.all_track.one_frame_pass()
