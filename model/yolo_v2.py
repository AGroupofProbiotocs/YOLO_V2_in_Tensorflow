import inspect
import os
import numpy as np
import tensorflow as tf
import time
from model.darknet import DarkNet448
from model.layers import *
from config import FLAGS
from utils.network_utils import cal_feat_size

class YOLO_V2:
    def __init__(self, base_model='darknet448'):
        if base_model == 'darknet448':
            self.base_model = DarkNet448()
        else:
            raise  ValueError('Unsopported base model for now!')
        self.anchors = FLAGS.anchors
        self.num_class = FLAGS.num_class
        self.num_anchor = len(self.anchors)

    def forward(self, input, is_training=False, reuse=False, name='yolo_v2'):
        '''
        The forward progress of YOLO_v2.
        :param input: a placeholder of input image
        :param reuse: whether to reuse the variables
        :param is_training: is training or not
        :param name: network name
        :return: pred_result: a tuple containing (pred_bboxes, pred_confidence, pred_offset, pred_class_prob)
        '''
        with tf.variable_scope(name, reuse=reuse):
            _ = self.base_model.forward(input, is_training=is_training)
            self.conv_17 = self.base_model.all_layers['convolutional_17']
            self.shortcut_18 = conv2d_layer(self.conv_17, 1, 64, is_training=is_training, name='shortcut_18')
            self.shortcut_19 = tf.space_to_depth(self.shortcut_18, 2, name='shortcut_19') #shape:(batch_size, 13, 13, 256)

            self.conv_23 = self.base_model.all_layers['convolutional_23']
            self.conv_24 = conv2d_layer(self.conv_23, 3, 1024, is_training=is_training, name='convolutional_24')
            self.conv_25 = conv2d_layer(self.conv_24, 3, 1024, is_training=is_training, name='convolutional_25')
            self.conv_26 = conv2d_layer(self.conv_25, 3, 1024, is_training=is_training, name='convolutional_26') #shape:(batch_size, 13, 13, 1024)

            self.concat_27 = tf.concat([self.shortcut_19, self.conv_26], axis=-1, name='concatenate_27') #shape:(batch_size, 13, 13, 1280)
            self.conv_28 = conv2d_layer(self.concat_27, 3, 1024, is_training=is_training, name='convolutional_28')  # shape:(batch_size, 13, 13, 1024)
            self.conv_29 = conv2d_layer(self.conv_28, 1, self.num_anchor*(self.num_class + 5), is_training=is_training, name='convolutional_29')  #shape:(batch_size, 13, 13, 125)

            self.input_shape = tf.shape(input)
            self.feat_shape = tf.shape(self.conv_29)

            pred_result = self.head(self.conv_29)

        return pred_result

    def head(self, feature):
        '''
        Convert the raw yolo output into separated predictions, i.e., pred_bboxes, pred_confidence and pred_class_prob,
        which are going to used in the loss calculation
        :param feature: the raw output of yolo_v2 network
        :return: pred_bboxes: predicted bbox conner coordinates with the shape of (batch_size, feat_h, feat_w, 5, 4)
                 pred_confidence: predicted confidence with the shape of (batch_size, feat_h, feat_w, 5, 1)
                 pred_confidence: predicted confidence with the shape of (batch_size, feat_h, feat_w, 5, 1)
                 pred_class_prob: predicted class probability with the shape of (batch_size, feat_h, feat_w, 5, 20)
        '''
        anchors = tf.constant(self.anchors, dtype=tf.float32)  #shape: (5, 2)
        anchors_w = tf.reshape(anchors[:, 0], (1, 1, 1, self.num_anchor)) #shape: (1, 1, 1, 5)
        anchors_h = tf.reshape(anchors[:, 1], (1, 1, 1, self.num_anchor))

        feat_h = tf.cast(self.feat_shape[1], 'float32')
        feat_w = tf.cast(self.feat_shape[2], 'float32')
        feature = tf.reshape(feature, (-1, feat_h, feat_w, self.num_anchor, self.num_class + 5))  #shape: (batch_size, feat_h, feat_w, 5, 25)

        # the first four channels are [center_x_offset, center_y_offset, w_offset, h_offset] respectively,
        # in which the predicted offset_x and offset_y are the shift values to the corresponding feature grid,
        # and the predicted w_offset and h_offset are actually the log value of the original bbox width and height.
        offset_x = feature[..., 0]  #shape: (batch_size, feat_h, feat_w, 5)
        offset_y = feature[..., 1]
        offset_w = feature[..., 2]
        offset_h = feature[..., 3]
        pred_offset = tf.stack([tf.nn.sigmoid(offset_x), tf.nn.sigmoid(offset_y), offset_w, offset_h], axis=-1) #shape: (batch_size, feat_h, feat_w, 5, 4)
        pred_confidence = tf.nn.sigmoid(feature[..., 4:5]) #shape: (batch_size, feat_h, feat_w, 5, 1)
        pred_class_prob = tf.nn.softmax(feature[..., 5:]) #shape: (batch_size, feat_h, feat_w, 5, 20)

        h_idx = tf.range(feat_h)
        w_idx = tf.range(feat_w)
        grid_x, grid_y = tf.meshgrid(w_idx, h_idx)
        grid_x = grid_x[None, :, :,None] # shape: (1, feat_h, feat_w, 1)
        grid_y = grid_y[None, :, :,None] # shape: (1, feat_h, feat_w, 1)

        # restore the offsets to the center coordinates, width and height of bboxes
        bbox_x = (tf.nn.sigmoid(offset_x) + grid_x) / feat_w  #shape: (batch_size, feat_h, feat_w, 5)
        bbox_y = (tf.nn.sigmoid(offset_y) + grid_y) / feat_h
        bbox_w = (tf.exp(offset_w) * anchors_w) / feat_w
        bbox_h = (tf.exp(offset_h) * anchors_h) / feat_h

        corner_x_min = bbox_x - 0.5 * bbox_w  # the coordinate value is from 0 to 1
        corner_y_min = bbox_y - 0.5 * bbox_h
        corner_x_max = bbox_x + 0.5 * bbox_w
        corner_y_max = bbox_y + 0.5 * bbox_h

        # the purpose of converting the predicted offset into bbox coordinate is to use it in the iou loss calculation
        pred_bboxes = tf.stack([corner_x_min, corner_y_min, corner_x_max, corner_y_max], axis=-1)

        return pred_bboxes, pred_confidence, pred_offset, pred_class_prob

    def loss(self, true_bboxes, matched_true_offset, matched_true_class_prob, target_anchor_mask,
             pred_bboxes, pred_offset, pred_confidence, pred_class_prob,
             threshold=0.6, rescore_confidence=True):
        '''
        YOLO_v2 loss, which composed by two loss: background loss (i.e., non-object loss) and object loss,
        in which object loss is further composed by confidence loss, offset loss and classification loss.
        
        :param true_bboxes: normalized true bboxes, shape: (batch_size, #objects, 4)
        :param target_anchor_mask: shape: (batch_size, feat_h, feat_w, #anchors, 1), #anchors = 5 for yolo_v2
        :param matched_true_offset: shape: (batch_size, feat_h, feat_w, #anchors, 4)
        :param matched_true_class_prob: shape: (batch_size, feat_h, feat_w, #anchors, 20)
        :param pred_offset: shape: (batch_size, feat_h, feat_w, #anchors, 4)
        :param pred_bboxes: shape: (batch_size, feat_h, feat_w, #anchors, 4)
        :param pred_confidence: shape: (batch_size, feat_h, feat_w, #anchors, 1)
        :param pred_class_prob: shape: (batch_size, feat_h, feat_w, #anchors, 20)
        :return: total_loss
        '''
        
        # the scale coefficient of each part of the loss 
        lambda_bg = 1.
        lambda_conf = 5.   
        lambda_offset = 1.
        lambda_class = 1.
        
        # calculate the background loss.
        # reshape the gt bbox and pred bbox for the convenience of iou calculation.
        true_bboxes = true_bboxes[:, None, None, :, None, :] #shape: (batch_size, 1, 1, #objects, 1, 4)
        pred_bboxes = pred_bboxes[:, :, :, None, :, :] #shape: (batch_size, feat_h, feat_w, 1, #anchors, 4)
        
        # calculate the iou
        min_corner = tf.maximum(true_bboxes[..., :2], pred_bboxes[..., :2]) #shape: (batch_size, feat_h, feat_w, #objects, #anchors, 2)
        max_corner = tf.minimum(true_bboxes[..., 2:], pred_bboxes[..., 2:]) #shape: (batch_size, feat_h, feat_w, #objects, #anchors, 2)
        inter_wh = tf.maximum(max_corner - min_corner, 0)
        area_inter = tf.reduce_prod(inter_wh, axis=-1)  #shape: (batch_size, feat_h, feat_w, #objects, #anchors)
        
        area_a = tf.reduce_prod(true_bboxes[..., 2:] - true_bboxes[..., :2], axis=-1)
        area_b = tf.reduce_prod(pred_bboxes[..., 2:] - pred_bboxes[..., :2], axis=-1)
        
        iou = area_inter / (area_a + area_b - area_inter) #shape: (batch_size, feat_h, feat_w, #objects, #anchors)

        # Create the bg mask, note that here we don't concern about which object the anchors are matched to. Since the
        # non-tergeted predicted bbox will be ignored, as long as the max iou between itself and a certain gt bbox are
        # larger than the threshold.
        max_iou = tf.reduce_max(iou, axis=3)[..., None] #shape: (batch_size, feat_h, feat_w, #anchors, 1)
        bg_mask = tf.cast(max_iou < threshold, dtype=tf.float32)  # only concern the sample with iou < threshold
        
        non_obj_loss = lambda_bg * tf.reduce_sum((1 - target_anchor_mask) * bg_mask * tf.square(pred_confidence), axis=[1, 2, 3, 4])

        # calculate the object loss
        # first term, the confidence loss. if the rescore flag is True, gt confidence will be set to be the iou.
        if rescore_confidence:
            true_confidence = max_iou
        else:
            true_confidence = 1
        conf_loss = lambda_conf * tf.reduce_sum(target_anchor_mask * tf.square(true_confidence - pred_confidence), axis=[1, 2, 3, 4])

        # second term, the offset loss.
        offset_loss = lambda_offset * tf.reduce_sum(target_anchor_mask * tf.square(matched_true_offset - pred_offset), axis=[1, 2, 3, 4])

        # third term, the class prob loss.
        class_loss = lambda_class * tf.reduce_sum(target_anchor_mask * tf.square(matched_true_class_prob - pred_class_prob), axis=[1, 2, 3, 4])

        obj_loss = conf_loss + offset_loss + class_loss

        total_loss = 0.5 * tf.reduce_mean(non_obj_loss + obj_loss)

        return total_loss

    def eval(self, pred_bbox, pred_conf, pred_prob, filter_threshold=0.7, nms_threshold=0.5, max_bbox_num=10):
        '''
        Filter the predicted result from the yolo_head and output the final prediction for evaluation, but note that
        the function only accepts the result of one sample as input for once, since the Tensor doesn't support the
        cyclic assignment.
        :param pred_bbox: the predicted bbox for a single input image, which is a tensor with shape: (1, feat_h, feat_w, 5, 4)
        :param pred_conf: the predicted confidence for a single input image, which is a tensor with shape: (1, feat_h, feat_w, 5, 1)
        :param pred_prob: the predicted class prob for a single input image, which is a tensor with shape: (1, feat_h, feat_w, 5, 20)
        :param filter_threshold: the threshold for filtering the predicted score to abandon the invalid predictions
        :param nms_threshold: the threshold to suppress the non-maximum
        :param max_bbox_num: the max bbox number to keep after nms
        :return:
        '''

        pred_score_all = pred_conf * pred_prob  #shape: (1, feat_h, feat_w, 5, 20)
        pred_score = tf.reduce_max(pred_score_all, axis=-1)  #keep the max score, #shape: (1, feat_h, feat_w, 5)
        pred_class = tf.argmax(pred_score_all, axis=-1)  #the index of the max score is the pred class, shape: (1, feat_h, feat_w, 5)
        bbox_test = pred_bbox

        bool_mask = (pred_score >= filter_threshold)  # the boolean mask used to keep the valid prediction, #shape: (1, feat_h, feat_w, 5)
        # input with an N-d tensor and a K-d mask, the 'tf.boolean_mask' function will output a (N-K+1)-d result.
        pred_bbox = tf.boolean_mask(pred_bbox, bool_mask) #shape: (?, 4)
        pred_score = tf.boolean_mask(pred_score, bool_mask) #shape: (?, )
        pred_class = tf.boolean_mask(pred_class, bool_mask) #shape: (?, )

        img_h = self.input_shape[1]
        img_w = self.input_shape[2]
        restore_scale = tf.cast(tf.stack([img_w, img_h, img_w, img_h]), 'float32')
        pred_bbox = pred_bbox * restore_scale # restore the coordinates in terms of the input

        # interchange the x and y coordinates, since the 'tf.image.non_max_suppression' accepts bbox like [y1, x1, y2, x2]
        pred_bbox_yx = tf.stack([pred_bbox[:, 1], pred_bbox[:, 0], pred_bbox[:, 3], pred_bbox[:, 2]], axis=-1) # shape: (?, 4)
        keep_index = tf.image.non_max_suppression(pred_bbox_yx, pred_score, max_bbox_num, nms_threshold) # shape: (?, )

        pred_bbox = tf.gather(pred_bbox, keep_index) #shape: (?, 4)
        pred_score = tf.gather(pred_score, keep_index) #shape: (?, )
        pred_class = tf.gather(pred_class, keep_index) #shape: (?, )

        return bbox_test, pred_bbox, pred_score, pred_class