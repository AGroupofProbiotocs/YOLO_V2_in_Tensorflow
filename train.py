import inspect
import os
import numpy as np
import tensorflow as tf
import math
import time
import csv
from datetime import datetime
from model.yolo_v2 import YOLO_V2
from utils.bbox_utils import *
from utils.eval_utils import *
from utils.vis_utils import *
from utils.train_utils import Save_Manager
from data.batch_generator import generate_train_batch, generate_val_batch
from data.load_data import *
from config import FLAGS


#--------------------------------------------------config tuning----------------------------------------------
CKPT_PATH = FLAGS.save_path_1
NUM_ANCHORS = len(FLAGS.anchors)

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPUs

if type(FLAGS.metrics) == str:
    METRICS = [FLAGS.metrics]
elif type(FLAGS.metrics) != list:
    raise ValueError('Unacceptable metrics type!')

def train():
    #--------------------------------------------------Data----------------------------------------------------
    train_id_list = get_id_list(FLAGS.voc_doc_path, file_name='trainval')
    # train_id_list = train_id_list[:56] # for debug
    train_steps = int(np.ceil(len(train_id_list) / FLAGS.batch_size_train))

    val_id_list = get_id_list(FLAGS.voc_doc_path, file_name='test')  # it's actually a validation set
    # val_id_list = val_id_list[:20] #for debug
    val_steps = len(val_id_list)  # the batch size for validation is always 1

    # generator of training set
    train_generator = generate_train_batch(FLAGS.voc_doc_path,
                                         train_id_list,
                                         anchors = FLAGS.anchors,
                                         batch_size = FLAGS.batch_size_train,
                                         img_size = FLAGS.image_size_train,
                                         feat_size = FLAGS.feature_size_train,
                                         random_shuffle=True,
                                         preprocess=True,
                                         augment=True)

    # generator of validation set
    val_generator = generate_val_batch(FLAGS.voc_doc_path,
                                       val_id_list,
                                       random_shuffle=False,
                                       preprocess=True,
                                       augment=False)

    #---------------------------------------------------Network------------------------------------------------
    # define palceholder
    img_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    true_bboxes = tf.placeholder(dtype=tf.float32, shape=[None, None, 4])
    true_offset = tf.placeholder(dtype=tf.float32, shape=[None, None, None, NUM_ANCHORS, 4])
    true_class_prob = tf.placeholder(dtype=tf.float32, shape=[None, None, None, NUM_ANCHORS, FLAGS.num_class])
    target_anchor_mask = tf.placeholder(dtype=tf.float32, shape=[None, None, None, NUM_ANCHORS, 1])
    is_training = tf.placeholder(dtype=tf.bool)

    yolo_v2 = YOLO_V2()
    pred_bboxes, pred_confidence, pred_offset, pred_class_prob = yolo_v2.forward(img_input, is_training)
    #TODO: calculate model loss on CPU to reduce GPU memory usage.
    loss = yolo_v2.loss(true_bboxes, true_offset, true_class_prob, target_anchor_mask,
                        pred_bboxes, pred_offset, pred_confidence, pred_class_prob,
                        threshold=FLAGS.bg_threshold, rescore_confidence=FLAGS.rescore_confidence)

    bbox_test, pred_bbox_val, pred_score_val, pred_class_val = yolo_v2.eval(pred_bboxes, pred_confidence, pred_class_prob,
                                                                 FLAGS.filter_threshold, FLAGS.nms_threshold,
                                                                 FLAGS.max_bbox_num)


    print('Successfully build the graph!')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    decayed_learning_rate = tf.train.exponential_decay(FLAGS.init_lr,
                                                       global_step,
                                                       FLAGS.lr_decay_epochs * train_steps,
                                                       FLAGS.lr_decay_rate,
                                                       staircase=True)  # lerning rate decay

    # execute update_ops to update batch_norm weights
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.GradientDescentOptimizer(decayed_learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list=tf.global_variables())  # 实例化保存器
    save_manager = Save_Manager(saver, CKPT_PATH, FLAGS.patient, FLAGS.metrics, mode=FLAGS.monitor_mode,
                                save_best=FLAGS.save_best, early_stop=FLAGS.early_stop)

    #-----------------------------------------------------Train------------------------------------------------
    with tf.Session() as sess:
        print('\nStart Training!')
        start_time = time.time()
        sess.run(init)
        record = [['Train loss', 'mAP']]

        for epoch in range(FLAGS.num_epoch):
            print(
                '\n==================================== {} - Epoch:{}/{} ===================================='.format(
                    str(datetime.now())[:-7], epoch, FLAGS.num_epoch))
            # training
            total_train_loss = 0
            for step in range(train_steps):
                img_train, true_bboxes_train, true_offsets_train, true_class_probs_train, anchor_masks_train = next(train_generator)

                _, train_loss = sess.run([train_op, loss], feed_dict={img_input: img_train,
                                                                      true_bboxes: true_bboxes_train,
                                                                      true_offset: true_offsets_train,
                                                                      true_class_prob: true_class_probs_train,
                                                                      target_anchor_mask: anchor_masks_train,
                                                                      is_training: True})

                total_train_loss += train_loss

                if (step+1) % FLAGS.bar_display_interval == 0:
                    view_bar(step, train_steps, extra_msg={'Current loss':train_loss})

            mean_train_loss = total_train_loss / train_steps

            # validation
            true_bboxes_val, true_labels_val = [], []
            pred_bboxes_val, pred_labels_val, pred_probs_val = [], [], []
            for step in range(val_steps):
                img_val, gt_bbox_val, gt_label_val = next(val_generator)

                pred_bbox, pred_label, pred_score = sess.run([pred_bbox_val, pred_score_val, pred_class_val],
                                                             feed_dict={img_input: img_val,
                                                                        is_training: False})

                true_bboxes_val.append(gt_bbox_val)
                true_labels_val.append(gt_label_val)
                pred_bboxes_val.append(pred_bbox)
                pred_labels_val.append(pred_label)
                pred_probs_val.append(pred_score)

                if step % FLAGS.img_display_interval == 0:
                    show_image(img_val[0], pred_bbox, pred_label, pred_score, FLAGS.minimum_threshold_vis)

            # print('\n', pred_bbox)
            AP, mAP = result_eval(pred_bboxes_val, pred_labels_val, pred_probs_val, true_bboxes_val, true_labels_val, use_07_metric=True)

            print(
                '\n===================================== {} - Validation ===================================='.format(
                    str(datetime.now())[:-7]) +
                '\nTraining loss: %.5f' % mean_train_loss +
                '\nValidation mAP: %.5f' % mAP)

            record.append([mean_train_loss, mAP])

            # use the save_manager to decide the model saving process
            current_value = []
            for each in METRICS:
                if each == 'training loss':
                    current_value.append(mean_train_loss)
                elif each == 'validation mAP':
                    current_value.append(mAP)
                else:
                    raise ValueError('Unknown metrics!')

            save_manager.run(sess, current_value, epoch)

            if save_manager.variable_dict['stop_training'] is True:
                print('Early stop!')
                break

        if not FLAGS.save_best:
            saver.save(sess, CKPT_PATH)
            print("Model saved in file: %s" % CKPT_PATH)

        # a brief report at the end of training
        end_time = time.time()
        print('The total training time is: %.2fmin' % ((end_time-start_time)/60.))
        for each in METRICS:
            print('The %s %s is obtained in epoch %s, the value is: %.5f' %
                  (FLAGS.monitor_mode, each, save_manager.variable_dict[each]['epoch'],
                   save_manager.variable_dict[each]['%s_value' % FLAGS.monitor_mode]))

        # save the training record
        with open(FLAGS.csv_record_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(record)


if __name__ == '__main__':
    tf.reset_default_graph()
    train()