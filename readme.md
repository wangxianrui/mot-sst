
# dataset
    dataset
        MOT17
        --train
        --test
# config.py
    you can modify all the hyper parameters in config.py
        CommomConfig
        TrainConfig
        EvalConfg
          
 
# training
    > python train_mot17.py
        auto save ckpt to checkpoints


# evaluation
    > python eval_mot17.py --type train
        create train dataset result to restult/train/txt

    > python eval_mot17.py --type test
        create test dataset result to restult/train/txt

# tool:
    > python tool/show_track.py --type train
    > python tool/show_track.py --type test
        create video from txt

    > python tool/show_gt.py
        create ground_truth video
        
    > python tool/performance.py
        calculate mota and motp from resutl and ground_truth
        
    > python tool/get_detection.py
        get detection txt from dataset, and then use show_track to show the detection
        
    > python tool/create_clean_detection.py
        get clean detection from dataset, iou confidence



