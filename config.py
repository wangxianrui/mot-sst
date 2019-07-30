# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""


class Config:
    data_root = '../dataset/MOT17'
    log_dir = 'logs'
    ckpt_dir = 'checkpoints'
    use_cuda = False
    mean_pixel = (127.5, 127.5, 127.5)
    sst_dim = 900
    max_object = 45
    image_size = (sst_dim, sst_dim)
    false_constant = 1
    detector = 'FRCNN'  # DPM, SDP, FRCNN


class TrainConfig(Config):
    backbone = 'pretrained/vgg16_reducedfc.pth'
    # backbone = 'pretrained/resnet50_reducedfc.pth'
    from_training = ''
    batch_size = 1
    num_workers = 8
    min_visibility = 0.3
    min_gap_frame = 0
    max_gap_frame = 10
    lr_init = 1e-3
    lr_map = {
        '1': 1e-2,
        '10': 1e-3,
        '15': 1e-4,
        '18': 1e-5,
    }
    max_epoch = 20
    momentum = 0.9
    weight_decay = 5e-4
    log_setp = 50
    save_step = 200


class EvalConfig(Config):
    model_path = 'checkpoints/sst900_0_19.pth'
    result_dir = 'result'
    max_track_frame = 10
