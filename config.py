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
    max_object = 10
    image_size = (sst_dim, sst_dim)
    false_constant = 1
    # DPM, SDP, FRCNN select detector in mot17 dataset
    detector = ''


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
    model_path = 'pretrained/sst900_mot.pth'
    result_dir = 'result'
    max_track_frame = 10
    valid_label = [3, 68]  # 3 car, 68 phone
    # confidence, low filter out, high keep
    low_confidence = 0.3
    high_confidence = 0.8
    # max interval frames, and min duration
    max_interval = 50
    min_duration = 125
