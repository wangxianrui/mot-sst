# ==========================================================================
#
# This file is a part of implementation for paper:
# DeepMOT: A Differentiable Framework for Training Multiple Object Trackers.
# This contribution is headed by Perception research team, INRIA.
#
# Contributor(s) : Yihong Xu
# INRIA contact  : yihong.xu@inria.fr
#
# ===========================================================================

import os
import cv2
import argparse
import numpy as np
import copy
import csv
import motmetrics
from config import EvalConfig as Config


def main(args):
    mh = motmetrics.metrics.create()
    data_root = os.path.join(Config.data_root, args.type)
    txts_dir = os.path.join(Config.result_dir, args.type, 'txt')
    txts_list = os.listdir(txts_dir)

    total_fn = 0
    total_fp = 0
    total_idsw = 0
    total_num_objects = 0
    total_matched = 0
    sum_distance = 0

    for txt_file in txts_list:
        vname = txt_file[:-4]
        print(vname)
        if not os.path.exists(os.path.join(data_root, vname, "gt/gt.txt")):
            print('error {} have no ground truth file in {}'.format(vname, data_root))

        acc = motmetrics.MOTAccumulator(auto_id=True)

        frames_gt = read_txt_gtV2(os.path.join(data_root, vname, "gt/gt.txt"))
        h, w, _ = cv2.imdecode(np.fromfile(os.path.join(data_root, vname, 'img1/000001.jpg'), dtype=np.uint8), -1).shape
        # h, w, _ = cv2.imread(os.path.join(data_root, vname, 'img1/000001.jpg')).shape
        frames_prdt = read_txt_predictionV2(os.path.join(txts_dir, txt_file))

        # evaluations
        for frameid in frames_gt.keys():
            gt_bboxes = np.array(frames_gt[frameid], dtype=np.float32)
            gt_ids = gt_bboxes[:, 0].astype(np.int32).tolist()
            if frameid in frames_prdt.keys():
                # get id track
                id_track = np.array(frames_prdt[frameid])[:, 0].astype(np.int32).tolist()
                # get a binary mask from IOU, 1.0 if iou < 0.5, else 0.0
                mask_IOU = np.zeros((len(frames_prdt[frameid]), len(frames_gt[frameid])))
                # distance matrix
                distance_matrix = []
                for i, bbox in enumerate(frames_prdt[frameid]):
                    iou = bb_fast_IOU_v1(bbox, frames_gt[frameid])
                    # threshold
                    th = np.zeros_like(iou)
                    th[np.where(iou <= args.threshold)] = 1.0
                    mask_IOU[i, :] = th
                    # distance
                    distance_matrix.append(1.0 - iou)

                distance_matrix = np.vstack(distance_matrix)
                distance_matrix[np.where(mask_IOU == 1.0)] = np.nan
                acc.update(gt_ids, id_track, np.transpose(distance_matrix))
            else:
                acc.update(gt_ids, [], [[], []])

        summary = mh.compute(acc, metrics=['motp', 'mota', 'num_false_positives', 'num_misses',
                                           'num_switches', 'num_objects', 'num_matches'], name='final')
        total_fp += float(summary['num_false_positives'].iloc[0])
        total_fn += float(summary['num_misses'].iloc[0])
        total_idsw += float(summary['num_switches'].iloc[0])
        total_num_objects += float(summary['num_objects'].iloc[0])
        total_matched += float(summary['num_matches'].iloc[0])
        sum_distance += float(summary['motp'].iloc[0]) * float(summary['num_matches'].iloc[0])
        strsummary = motmetrics.io.render_summary(
            summary, formatters={'mota': '{:.2%}'.format},
            namemap={'motp': 'MOTP', 'mota': 'MOTA', 'num_false_positives': 'FP',
                     'num_misses': 'FN', 'num_switches': "ID_SW", 'num_objects': 'num_objects'}
        )
        print(strsummary)

    print("avg mota: {:.3f} %".format(100.0 * (1.0 - (total_idsw + total_fn + total_fp) / total_num_objects)))
    print("avg motp: {:.3f} %".format(100.0 * (1.0 - sum_distance / total_matched)))
    print("total fn: ", total_fn)
    print("total fp: ", total_fp)
    print("total idsw: ", total_idsw)
    print("total_num_objects: ", total_num_objects)


def reorder_frameID(frame_dict):
    """
    reorder the frames dictionary in a ascending manner
    :param frame_dict: a dict with key = frameid and value is a list of lists [object id, x, y, w, h] in the frame, dict
    :return: ordered dict by frameid
    """
    keys_int = sorted([int(i) for i in frame_dict.keys()])

    new_dict = {}
    for key in keys_int:
        new_dict[str(key)] = frame_dict[str(key)]
    return new_dict


def xywh2xyxy(bbox):
    """
    convert bbox from [x,y,w,h] to [x1, y1, x2, y2]
    :param bbox: bbox in string [x, y, w, h], list
    :return: bbox in float [x1, y1, x2, y2], list
    """
    copy.deepcopy(bbox)
    bbox[0] = float(bbox[0])
    bbox[1] = float(bbox[1])
    bbox[2] = float(bbox[2]) + bbox[0]
    bbox[3] = float(bbox[3]) + bbox[1]

    return bbox


def bb_fast_IOU_v1(boxA, boxB):
    """
    Calculation of IOU, version numpy
    :param boxA: numpy array [top left x, top left y, x2, y2]
    :param boxB: numpy array of [top left x, top left y, x2, y2], shape = [num_bboxes, 4]
    :return: IOU of two bounding boxes of shape [num_bboxes]
    """
    if type(boxA) is type([]):
        boxA = np.array(copy.deepcopy(boxA), dtype=np.float32)[-4:]
        boxB = np.array(copy.deepcopy(boxB), dtype=np.float32)[:, -4:]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[0], boxB[:, 0])
    yA = np.maximum(boxA[1], boxB[:, 1])
    xB = np.minimum(boxA[2], boxB[:, 2])
    yB = np.minimum(boxA[3], boxB[:, 3])

    # compute the area of intersection rectangle
    interArea = np.maximum(0.0, xB - xA + 1) * np.maximum(0.0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def read_txt_gtV2(textpath):
    """
    read gt.txt to a dict
    :param textpath: text path, string
    :return: {frame_id : [[object id, x1, y1, x2, y2],...], ...}
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        frames = {}
        for line in f_csv:
            if len(line) == 1:
                line = line[0].split(',')
            # we only consider "pedestrian" class #
            if len(line) < 7 or int(line[7]) != 1 or int(line[6]) == 0:
                continue
            index = str(int(float(line[0])))
            if index not in frames:
                frames[index] = []
            bbox = xywh2xyxy(line[2:6])
            frames[index].append([int(float(line[1]))] + bbox)
    ordered = reorder_frameID(frames)
    return ordered


def read_txt_predictionV2(textpath):
    """
    read prediction text file to a dict
    :param textpath: text path, String
    :return: a dict with key = frameid and value is a list of lists [track_id, x1, y1, x2, y2] in the frame, dict
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        frames = {}
        for line in f_csv:
            if len(line) == 1:
                line = line[0].split(' ')
            if len(line) <= 5:
                continue
            index = str(int(float(line[0])))
            if index not in frames:
                frames[index] = []
            bbox = xywh2xyxy(line[2:6])
            frames[index].append([int(float(line[1]))] + bbox)
    ordered = reorder_frameID(frames)
    return ordered


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.5, type=float, help='distance matrix threshold')
    parser.add_argument('--type', required=True, type=str, help='train or test')
    args = parser.parse_args()
    main(args)
