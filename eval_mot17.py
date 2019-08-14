'''
@Author: rayenwang
@Date: 2019-08-14 16:30:02
@Description: 
'''

import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from config import EvalConfig as Config
from pipline.mot_eval_dataset import MOTEvalDataset
from network.tracker import SSTTracker


def eval(args):
    if not os.path.exists(os.path.join(Config.result_dir, args.type, 'txt')):
        os.makedirs(os.path.join(Config.result_dir, args.type, 'txt'))

    video_list = os.listdir(os.path.join(Config.data_root, args.type))
    for vname in video_list:
        if Config.detector not in vname:
            continue

        img_dir = os.path.join(Config.data_root, args.type, vname, 'img1')
        det_file = os.path.join(Config.data_root, args.type, vname, 'det/det.txt')
        res_file = os.path.join(Config.result_dir, args.type, 'txt', vname + '.txt')
        result = list()

        print('start processing {}'.format(res_file))
        tracker = SSTTracker()
        dataset = MOTEvalDataset(image_folder=img_dir, detection_file_name=det_file)
        dataset_iter = iter(dataset)
        for i in tqdm(range(len(dataset))):
            img, det, mask, h, w = next(dataset_iter)
            # ## test
            # import cv2
            # if img is not None:
            #     image = img.permute(1, 2, 0).clone().numpy()
            #     for d in det[index].clone().numpy():
            #         x, y, w, h, _ = d * 900
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
                mask = mask.cuda()
            print(i)

            # track
            tracker.update(img, det, mask, i)

            # save result
            for t in tracker.tracks:
                n = t.nodes[-1]
                if t.age == 1:
                    b = n.get_box(i, tracker.recorder)
                    # -1 for x, y, z
                    result.append([i + 1] + [t.id] + [b[0] * w, b[1] * h, b[2] * w, b[3] * h] + [b[4], -1, -1, -1])
        np.savetxt(res_file, post_precessing(result), fmt='%.2f', delimiter=',')
        print('finished processing {}'.format(res_file))


def post_precessing(result):
    result = np.asarray(result).reshape(-1, 10)
    id_list = np.unique(result[:, 1])
    for track_id in id_list:
        track = result[result[:, 1] == track_id, :]
        if np.max(track[:, 6]) < Config.high_confidence:
            result = result[result[:, 1] != track_id, :]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='test', help='eval train or test dataset')
    args = parser.parse_args()
    with torch.no_grad():
        eval(args)
