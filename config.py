'''
@Author: rayenwang
@Date: 2019-07-17 17:34:56
@LastEditTime: 2019-07-19 18:45:11
@Description: 
'''


class Config:
    data_root = '../dataset/MOT17'
    log_dir = 'logs'
    ckpt_dir = 'checkpoints'
    use_cuda = True
    mean_pixel = (127.5, 127.5, 127.5)
    sst_dim = 900
    max_object = 80
    image_size = (sst_dim, sst_dim)
    false_constant = 5
    detector = 'FRCNN'  # DPM, SDP, FRCNN


class TrainConfig(Config):
    backbone = 'pretrained/vgg16_reducedfc.pth'
    from_training = 'checkpoints/sst900_19_599.pth'
    batch_size = 8
    num_workers = 8
    min_visibility = 0.3
    min_gap_frame = 0
    max_gap_frame = 15
    lr_init = 1e-2
    lr_decay = 0.1
    lr_epoch = [20, 30, 35, 38]
    max_epoch = 40
    momentum = 0.9
    weight_decay = 5e-4
    log_setp = 50
    save_step = 200


class EvalConfig(Config):
    model_path = 'checkpoints/sst900_19_599.pth'
    result_dir = 'result/'
    max_track_frame = 10
