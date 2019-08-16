'''
@Author: rayenwang
@Date: 2019-08-14 11:16:18
@Description: 
'''


class Config:
    data_root = '../dataset/movies_08_06'
    log_dir = 'logs'
    ckpt_dir = 'checkpoints'
    use_cuda = False
    mean_pixel = (127.5, 127.5, 127.5)
    sst_dim = 900
    max_object = 8
    image_size = (sst_dim, sst_dim)
    false_constant = 1
    detector = ''  # DPM, SDP, FRCNN


class TrainConfig(Config):
    backbone = 'pretrained/vgg16_reducedfc.pth'
    from_training = ''
    batch_size = 1
    num_workers = 1
    min_visibility = 0.3
    max_gap_frame = 15
    lr_init = 1e-3
    lr_map = {
        '1': 1e-2,
        '5': 1e-3,
        '8': 1e-4,
    }
    max_epoch = 10
    momentum = 0.9
    weight_decay = 5e-4
    focal_alpha = 1
    focal_gamma = 2
    log_setp = 10
    save_step = 200


class EvalConfig(Config):
    model_path = 'pretrained/sst900_final.pth'
    result_dir = 'result'
    max_track_frame = 5
    # filter out
    low_confidence = 0.3
    # add to track
    high_confidence = 0.8
    # iou
    iou_threshold = 0.3
    # max interval frames, and min duration
    max_interval = 50
    min_duration = 125
