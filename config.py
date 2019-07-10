class Config:
    '''common'''
    data_root = 'dataset/MOT17/'
    log_dir = 'logs/'
    ckpt_dir = 'checkpoints/'
    use_cuda = False
    mean_pixel = (104, 117, 123)
    sst_dim = 900
    max_object = 80
    base_net = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
    extra_net = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256, 128, 'S', 256, 128, 256]
    selector_size = (255, 113, 56, 28, 14, 12, 10, 5, 3)
    selector_channel = (60, 80, 100, 80, 60, 50, 40, 30, 20)
    final_net = [1040, 512, 256, 128, 64, 1]  # final_net[0] = np.sum(selector_channel) * 2
    vgg_source = [15, 25, -1]
    default_mbox = [4, 6, 6, 6, 4, 4]
    false_constant = 10
    '''train'''
    backbone = 'pretrained/vgg16_reducedfc.pth'
    batch_size = 1
    num_workers = 8
    detector = 'DPM'  # DPM, SDP, FRCNN
    min_visibility = 0.3
    min_gap_frame = 0
    max_gap_frame = 30
    lower_saturation = 0.7
    upper_saturation = 1.5
    lower_contrast = 0.7
    upper_contrast = 1.5
    max_expand = 1.2
    lr_init = 1e-2
    lr_decay = 0.1
    lr_epoch = [50, 80, 100, 110]
    max_epoch = 120
    momentum = 0.9
    weight_decay = 5e-4
    log_setp = 500
    save_step = 1000
    '''eval'''
    resume = 'pretrained/sst300_0712_83000.pth'
    # resume = 'checkpoints/sst300_0712_83000.pth'
    result_dir = 'result/'
