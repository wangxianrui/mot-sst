"""
# TODO
Config
TrainConfig(Config)
EvalConfig(Config)
"""

class Config:
    pass


class 


class Config:
    '''common'''
    data_root = '../dataset/MOT17'
    log_dir = 'logs'
    ckpt_dir = 'checkpoints'
    use_cuda = False
    mean_pixel = (127.5, 127.5, 127.5)
    sst_dim = 900
    max_object = 80
    base_net = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
    extra_net = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256, 128, 'S', 256, 128, 256]
    selector_size = (255, 113, 56, 28, 14, 12, 10, 5, 3)
    selector_channel = (60, 80, 100, 80, 60, 50, 40, 30, 20)
    final_net = [1040, 512, 256, 128, 64, 1]  # final_net[0] = np.sum(selector_channel) * 2
    vgg_source = [15, 25, -1]
    false_constant = 10
    '''train'''
    backbone = 'pretrained/vgg16_reducedfc.pth'
    from_training = ''
    batch_size = 1
    num_workers = 8
    detector = 'FRCNN'  # DPM, SDP, FRCNN
    min_visibility = 0.3
    min_gap_frame = 0
    max_gap_frame = 30
    lr_init = 1e-2
    lr_decay = 0.1
    lr_epoch = [50, 80, 100, 110]
    max_epoch = 120
    momentum = 0.9
    weight_decay = 5e-4
    log_setp = 1
    save_step = 1
    '''eval'''
    model_path = 'pretrained/sst900_final.pth'
    result_dir = 'result/'


class TrackerConfig:
    cuda = Config.use_cuda
    decay = 1.0
    image_size = (Config.sst_dim, Config.sst_dim)
    max_bad_node = 0.9
    max_draw_track_node = 30
    max_object = Config.max_object
    max_record_frame = 30
    max_track_age = 12
    max_track_node = 12
    mean_pixel = Config.mean_pixel
    min_iou = [0.3, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0]
    min_iou_frame_gap = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_merge_threshold = 0.9
    roi_verify_punish_rate = 0.6
    roi_verify_max_iteration = 6
    sst_model_path = Config.model_path
