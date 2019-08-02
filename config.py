# -*- coding:utf-8 -*-
"""
@authors: rayenwang
@time: ${DATE} ${TIME}
@file: ${NAME}.py
@description:
"""


class Config:
    data_root = '../dataset/CAR_tracking'
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
    # backbone = 'pretrained/resnet50_reducedfc.pth'
    from_training = ''
    batch_size = 1
    num_workers = 1
    min_visibility = 0.3
    min_gap_frame = 0
    max_gap_frame = 10
    lr_init = 1e-2
    lr_map = {
        '1': 1e-3,
        '5': 1e-4,
        '7': 1e-5,
    }
    max_epoch = 8
    momentum = 0.9
    weight_decay = 1e-5
    log_setp = 50
    save_step = 200


class EvalConfig(Config):
    model_path = 'pretrained/sst900_final.pth'
    result_dir = 'result'
    max_track_frame = 5
