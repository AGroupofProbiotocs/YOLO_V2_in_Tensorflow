import os
import numpy as np
import tensorflow as tf
import math
from .bbox_utils import *

def result_eval(pred_bboxes, pred_labels, pred_probs, gt_bboxes, gt_labels, gt_difficults=None, iou_thresh=0.5,
                use_07_metric=False):

    '''
        Evaluate the prediction result by calculate average precisions.

        :param pred_bboxes: A list with the length of the validation/test samples, where each element is an array with
                            the shape of (#pred_bbox, 4)
        :param pred_labels: A list with the length of the validation/test samples, where each element is an array with
                            the shape of (#pred_bbox, )
        :param pred_probs: A list with the length of the validation/test samples, where each element is an array with
                           the shape of (#pred_bbox, )
        :param gt_bboxes: A list with the length of the validation/test samples, where each element is an array with
                          the shape of (#object, 4)
        :param gt_labels: A list with the length of the validation/test samples, where each element is an array with
                          the shape of (#object, 4)
        :param difficults: A list with the length of the validation/test samples, where each element is a number of 1
                           or 0. 1 means the sample is not difficult to detect, 0 is the opposite. Default is None,
                           which means all the predicted samples will be treat as not difficult.
        :param iou_thresh: A number denotes the IoU threshold.
        :param use_07_metric: Whether to use PASCAL VOC 2007 evaluation metric for calculating average precision.
                              The default value is`False`.
    '''
    prec, rec = calc_prec_rec(pred_bboxes, pred_labels, pred_probs, gt_bboxes, gt_labels, gt_difficults, iou_thresh)

    num_cls = len(prec)
    ap = np.empty(num_cls)
    for i in range(num_cls):
        if prec[i] is None or rec[i] is None:
            ap[i] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[i] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[i] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[i])[rec[i] >= t])
                ap[i] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mprec = np.concatenate(([0], np.nan_to_num(prec[i]), [0]))
            mrec = np.concatenate(([0], rec[i], [1]))

            mprec = np.maximum.accumulate(mprec[::-1])[::-1]

            # to calculate area under PR curve, look for points where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[i] = np.sum((mrec[i + 1] - mrec[i]) * mprec[i + 1])

    map =np.nanmean(ap)

    return ap, map


def calc_prec_rec(pred_bboxes, pred_labels, pred_probs, gt_bboxes, gt_labels, difficults=None, iou_thresh=0.5):
    '''
    Calculate the precision and recall of predicted result. The code is based on the evaluation code provided
    in Vocdevkit document.

    :param pred_bboxes: A list with the length of the validation/test samples, where each element is an array with
                        the shape of (#pred_bbox, 4)
    :param pred_labels: A list with the length of the validation/test samples, where each element is an array with
                        the shape of (#pred_bbox, )
    :param pred_probs: A list with the length of the validation/test samples, where each element is an array with
                       the shape of (#pred_bbox, )
    :param gt_bboxes: A list with the length of the validation/test samples, where each element is an array with
                      the shape of (#object, 4)
    :param gt_labels: A list with the length of the validation/test samples, where each element is an array with
                      the shape of (#object, )
    :param difficults: A list with the length of the validation/test samples, where each element is a number of 1
                       or 0. 1 means the sample is not difficult to detect, 0 is the opposite. Default is None,
                       which means all the predicted samples will be treat as not difficult.
    :param iou_thresh: A number denotes the IoU threshold.
    :return: precision, recall. Array for drawing the precision-recall curve
    '''

    if difficults == None:
        difficults = [None]*len(pred_labels)

    obj_num = {} # the total object number of each class need to be detected, difficult objects are ignored by default
    match = {}  # matched samples, each object can only be detected for once, repeated detection is treated as nagative
    probs = {}  # prediction probability of the detection of each object.

    for (pred_bbox, pred_label, pred_prob, gt_bbox, gt_label, difficult) in \
        zip(pred_bboxes, pred_labels, pred_probs, gt_bboxes, gt_labels, difficults):

        if difficult == None:
            difficult = np.zeros_like(gt_label)
            
        # the "concatenate" and "unique" operations are used to extract the involved labels
        for i in np.unique(np.concatenate([pred_label, gt_label]).astype(int)): # i is the label in current image
            # initialize three dictionaries
            if i not in obj_num.keys():
                obj_num[i] = 0
            if i not in probs.keys():
                probs[i] = []
            if i not in match.keys():
                match[i] = []

            # select the bbox with current label
            pred_idx_i = (pred_label == i)
            pred_prob_i = pred_prob[pred_idx_i]
            pred_bbox_i = pred_bbox[pred_idx_i]
            # sort the samples by the order of prob, it makes sure that the prediction with highest prob will be used
            prob_oder = pred_prob_i.argsort()[::-1]
            pred_prob_i = pred_prob_i[prob_oder]
            pred_bbox_i = pred_bbox_i[prob_oder]

            gt_idx_i = (gt_label == i)
            gt_bbox_i = gt_bbox[gt_idx_i]
            difficult_i = difficult[gt_idx_i]

            # count the total objects
            obj_num[i] += np.sum(1 - difficult_i)
            # add elements to the list of probs of class i
            probs[i].extend(pred_prob_i)

            # if didn't classify this kind of object, i.e., fail to recognize the existed object
            # i.e., the label exists in gt, but do not exist in pred.
            if len(pred_bbox_i) == 0:
                # in this case, the probs[i] as well as the match[i] will add nothing
                continue
            # if the classified object don't exist in the image, i.e., don't match the gt label
            # i.e., the label exists in pred, but do not exist in gt.
            if len(gt_bbox_i) == 0:
                # set the corresponding flags in 'match' to be zeros
                match[i].extend([0] * pred_bbox_i.shape[0])
                continue

            '''
            As an example for "bbox_iou":
            For the current image, let i denote the class 'car', and the 'pred_bbox_i' be:       
                            [[0,  0,  20, 20], 
                             [45, 43, 90, 92], 
                             [42, 40, 88, 91],
                             [20, 21, 48, 50]]   #(4, 4)
                                                      
            And we assume that the 'gt_bbox_i' for 'car' is:     
                            [[0,  0,  20, 20], 
                             [45, 45, 90, 90]]   #(2, 4)
                             
            The the 'ious' would be:
                                [[1.0, 0.0],
                                 [0.0, 0.9],
                                 [0.0, 0.8],
                                 [0.0, 0.1]]     #(4, 2)  
            '''
            ious = bbox_iou(pred_bbox_i, gt_bbox_i)
            obj_index = np.argmax(ious, axis=1)  # an array like [0, 1, 1, 1]
            # if a certain iou is lower than the iou_thresh, treat it as a failure detection
            obj_index[np.max(ious, axis=1) < iou_thresh] = -1  #[0, 1, 1, -1]

            detected = np.zeros(gt_bbox_i.shape[0])
            for idx in obj_index:
                if idx == -1:
                    match[i].append(0)
                else:
                    if difficult_i[idx] == 1:
                        match[i].append(-1)   # by default, the difficult samples will be ignored
                    else:
                        if detected[idx] == 0:
                            match[i].append(1)
                        else:
                            match[i].append(0)  # repeated detection is treated as nagative

                    detected[idx] = 1  # means already detected
            # in terms of the example, the "match" list for the current class in the current image would be [1, 1, 0, 0]

    num_class = max(obj_num.keys()) + 1
    prec = [None] * num_class
    rec = [None] * num_class

    for i in obj_num.keys():   # i denotes the ith class
        score_i = np.array(probs[i])  # a array like [0.9, 0.8, 0.7, 0.01, 0.9, 0.5]
        match_i = np.array(match[i], dtype=np.int8)  # a array like [1, 1, 0, 0, 1, 0]

        #TODO: what's the difference if not sorted
        order = score_i.argsort()[::-1]
        match_i = match_i[order]

        tp = np.cumsum(match_i == 1)
        fp = np.cumsum(match_i == 0)

        # If an element of fp + tp is 0, the corresponding element of prec[i] is nan.
        prec[i] = tp / (fp + tp)
        # If obj_num[i] is 0, rec[i] is None. obj_num[i] = fp[-1] + tp[-1]
        if obj_num[i] > 0:
            rec[i] = tp / obj_num[i]

    return prec, rec



                    


