import os
from tqdm import trange

import numpy as np

from config import Config
from pipline.mot_eval_dataset import MOTEvalDataset
from network.tracker import SSTTracker, TrackerConfig

from tools.timer import Timer


def eval():
    result_dir = os.path.join(Config.result_dir, 'texts/')
    data_root = os.path.join(Config.data_root, 'test/')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    choice = (0, 0, 4, 0, 3, 3)
    TrackerConfig.set_configure(choice)
    video_list = os.listdir(data_root)
    timer = Timer()
    for vname in video_list:
        img_dir = '{}{}/img1'.format(data_root, vname)
        det_file = '{}{}/det/det.txt'.format(data_root, vname)
        res_txt = '{}{}.txt'.format(result_dir, vname)
        result = list()

        print('start processing {}'.format(res_txt))
        tracker = SSTTracker()
        dataset = MOTEvalDataset(image_folder=img_dir, detection_file_name=det_file, min_confidence=0.0)
        dataset_iter = iter(dataset)
        for i in trange(len(dataset)):
            item = next(dataset_iter)
            if not item:
                continue
            img = item[0]
            det = item[1]
            if img is None or det is None or len(det) == 0:
                continue

            # detection && track
            if len(det) > Config.max_object:
                det = det[:Config.max_object, :]
            h, w, _ = img.shape
            det[:, [2, 4]] /= float(w)
            det[:, [3, 5]] /= float(h)
            timer.tic()
            image_org = tracker.update(img, det[:, 2:6], False, i)
            timer.toc()

            # save result
            for t in tracker.tracks:
                n = t.nodes[-1]
                if t.age == 1:
                    b = n.get_box(tracker.frame_index - 1, tracker.recorder)
                    result.append([i] + [t.id] + [b[0] * w, b[1] * h, b[2] * w, b[3] * h] + [-1, -1, -1, -1])
        np.savetxt(res_txt, np.int_(result), fmt='%i')
        print('finished processing {}'.format(res_txt))
    print('total time {}'.format(timer.total_time))
    print('average time {}'.format(timer.average_time))


if __name__ == '__main__':
    eval()
