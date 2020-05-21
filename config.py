class Config():
    ## The following flags are related to data loading
    voc_doc_path = '/home/xianglong/Dataset/VOC2007/'  # The voc document path to load data

    ## The following flags define hyper-parameters regards training
    num_epoch = 60  # Total epochs that you want to train
    batch_size_train = 28  # Train batch size

    init_lr = 1e-3  # Initial learning rate#
    lr_decay_epochs = 9  # The learning rate decay interval
    lr_decay_rate = 0.5  # How much to decay the learning rate each time
    weight_decay = 0.0005  # scale for l2 regularization

    GPUs = '2'  # Which GPUs you want to used for training
    metrics = 'validation mAP'  # The metrics you want to monitor
    monitor_mode = 'max'  # the monitor mode corresponds to the metrics
    save_best = True  # Save the model when better metrics are achieved during training
    early_stop = False  # Stop training when the metrics didnt get improved for certain epochs
    patient = 20  # The maxinum unimproved epoch number you want to wait before early stop
    bar_display_interval = 2  # Steps interval to display the training progress
    img_display_interval = 20  # Steps interval to display the prediction bbox

    ## The following flags are related to image pre-processing and augmentation
    image_size_train = [448, 448]
    # parameters used in augmentation
    augment_chance = 0.5  # The change to peform each augmentation step
    intensity_change = 10.  # The intensity value to add to or sub from the image
    brightness_change = 0.15  # The percentage to enhance or worsen the image brigtness
    contrast_change = 0.15  # The percentage to enhance or worsen the image contrast
    min_edge = 5 # The minimum valid bbox edge for the augmented bbox

    ## The following flags are related to module config
    num_class = 20 # The class number
    anchors = [[0.57273, 0.677385], [1.87446, 2.06253],
               [3.33843, 5.47434], [7.88282, 3.52778],
               [9.77052, 9.16828]] # The predifined anchors provided by the yolo_v2 author
    feature_size_train = [14, 14]
    bg_threshold = 0.6   # The threshold to select the background(non-object) prediction samples
    rescore_confidence = True  # whether to rescore the confidence with the iou
    filter_threshold = 0.3  # The threshold for filtering the predicted score to abandon the invalid predictions
    nms_threshold = 0.5  # The threshold to suppress the non-maximum
    max_bbox_num = 10  # The max bbox number to keep after nms
    # parameters for prediction
    minimum_threshold_vis = 1.

    ## The setting and path for saving and restoring model
    save_path_1 = './checkpoints/1/model'  # Checkpoint directory to save
    save_path_2 = './checkpoints/2/model'  # Checkpoint directory to save
    save_path_3 = './checkpoints/3/model'  # Checkpoint directory to save
    vgg16_npy_path = './weights/vgg16.npy'
    csv_record_path = './result/record/loss_map_record.csv'  # CSV directory to save
    restore_path = './checkpoints/3/model'  # Checkpoint directory to restore
    # is_use_ckpt = False    #Whether to load a checkpoint and continue training


FLAGS = Config()