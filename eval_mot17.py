import os
import argparse
from tqdm import tqdm
import time
import numpy as np

from config import Config
from pipline.mot_eval_dataset import MOTEvalDataset
from network.tracker import SSTTracker, TrackerConfig


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

    choice = (0, 0, 4, 0, 3, 3)
    TrackerConfig.set_configure(choice)
    video_list = os.listdir(os.path.join(Config.data_root, args.type))
    timer = Timer()
    for vname in video_list:
        img_dir = os.path.join(Config.data_root, args.type, vname, 'img1')
        det_file = os.path.join(Config.data_root, args.type, vname, 'det/det.txt')
        res_file = os.path.join(Config.result_dir, args.type, 'txt', vname + '.txt')
        result = list()

        print('start processing {}'.format(res_file))
        tracker = SSTTracker()
        dataset = MOTEvalDataset(image_folder=img_dir, detection_file_name=det_file, min_confidence=0.0)
        dataset_iter = iter(dataset)
        for i in tqdm(range(len(dataset))):
            # TODO
            # return formated img, det, and ori_shape for output
            img, det = next(dataset_iter)
            if img is None or det is None or len(det) == 0:
                continue

            # TODO
            # move format to dataset
            # detection && track
            if len(det) > Config.max_object:
                det = det[:Config.max_object, :]
            h, w, _ = img.shape
            det[:, [2, 4]] /= float(w)
            det[:, [3, 5]] /= float(h)

            timer.tic()
            tracker.update(img, det[:, 2:6], i)
            timer.toc()

            # save result
            for t in tracker.tracks:
                n = t.nodes[-1]
                if t.age == 1:
                    b = n.get_box(tracker.frame_index - 1, tracker.recorder)
                    result.append([i] + [t.id] + [b[0] * w, b[1] * h, b[2] * w, b[3] * h] + [-1, -1, -1, -1])
        np.savetxt(res_file, np.int_(result), fmt='%i', delimiter=',')
        print('finished processing {}'.format(res_file))
    print('total time {}'.format(timer.total_time))
    print('average time {}'.format(timer.average_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='test', help='eval train or test dataset')
    args = parser.parse_args()
    eval(args)
