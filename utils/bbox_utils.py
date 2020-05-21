import os
import numpy as np
import tensorflow as tf
import math

def tune_bbox(anchors, coor_offset):
    '''
    tune the bboxes according to the coor_offset produced by RPN.

    :param anchors: with the shape of (#anchors, 4)
    :param coor_offset: with the shape of (#anchors, 4)
    :return: tuned_bbox
    '''
    tuned_bbox = np.empty_like(anchors)

    anchor_width = anchors[:, 2] - anchors[:, 0]
    anchor_height = anchors[:, 3] - anchors[:, 1]
    anchor_center_x = anchors[:, 0] + anchor_width
    anchor_center_y = anchors[:, 1] + anchor_height

    tuned_anchor_center_x = anchor_width * coor_offset[:, 0] + anchor_center_x
    tuned_anchor_center_y = anchor_height * coor_offset[:, 1] + anchor_center_y

    tuned_anchor_width = np.exp(coor_offset[:, 2]) * anchor_width
    tuned_anchor_height = np.exp(coor_offset[:, 3]) * anchor_height

    tuned_bbox[:, 0] = tuned_anchor_center_x - tuned_anchor_width / 2.
    tuned_bbox[:, 1] = tuned_anchor_center_y - tuned_anchor_height / 2.
    tuned_bbox[:, 2] = tuned_anchor_center_x + tuned_anchor_width / 2.
    tuned_bbox[:, 3] = tuned_anchor_center_y + tuned_anchor_height / 2.

    return tuned_bbox

def calc_offset(src_anchors, pred_anchors):
    coor_offset = np.empty_like(src_anchors)

    src_anchor_width = src_anchors[:, 2] - src_anchors[:, 0]
    src_anchor_height = src_anchors[:, 3] - src_anchors[:, 1]
    src_anchor_center_x = src_anchors[:, 0] + src_anchor_width
    src_anchor_center_y = src_anchors[:, 1] + src_anchor_height

    pred_anchor_width = pred_anchors[:, 2] - pred_anchors[:, 0]
    pred_anchor_height = pred_anchors[:, 3] - pred_anchors[:, 1]
    pred_anchor_center_x = pred_anchors[:, 0] + pred_anchor_width
    pred_anchor_center_y = pred_anchors[:, 1] + pred_anchor_height

    coor_offset[:, 0] = (pred_anchor_center_x - src_anchor_center_x) / src_anchor_width
    coor_offset[:, 1] = (pred_anchor_center_y - src_anchor_center_y) / src_anchor_height
    coor_offset[:, 2] = np.log(src_anchor_width / pred_anchor_width)
    coor_offset[:, 3] = np.log(src_anchor_height / pred_anchor_height)

    return coor_offset

# bbox_iou function accepting two single input, unused now.
# def bbox_iou(bbox1, bbox2):
#     '''
#     calculate the intersection over union of two bboxes.
#     :param bbox1: with the shape of (m, 4)
#     :param bbox2: with the shape of (n, 4)
#     :return: with the shape of (m, n)
#     '''
#     width1 = bbox1[2] - bbox1[0]
#     height1 = bbox1[3] -  bbox1[1]
#     bbox1_size = width1 *height1
# 
#     width2 = bbox2[2] - bbox2[0]
#     height2 = bbox2[3] - bbox2[2]
#     bbox2_size = width2 * height2
# 
#     inter_width = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
#     inter_height = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
#     if (inter_width>0) and (inter_height>0):
#         intersection = inter_width*inter_height
#         union = bbox1_size + bbox2_size - intersection
#         iou = intersection/(union + 1e-6)
#     else:
#         iou = 0
# 
#     return iou

def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (ndarray): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (ndarray): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        ndarray:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if len(bbox_a.shape) == 1:
        bbox_a = bbox_a[None, :]
    if len(bbox_b.shape) == 1:
        bbox_b = bbox_b[None, :]

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[None, :, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[None, :, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

    # union_wh = np.maximum(br - tl, 0)
    # area_i = union_wh[:, :, 0] * union_wh[:, :, 1]
    # a_wh = bbox_a[:, 2:] - bbox_a[:, :2]
    # area_a = a_wh[:, 0] * a_wh[:, 1]
    # b_wh = bbox_b[:, 2:] - bbox_b[:, :2]
    # area_b = b_wh[:, 0] * b_wh[:, 1]

    iou = area_i / (area_a[:, None] + area_b - area_i)

    return iou

def non_maximum_suppression(roi, threshold):
    '''
    non-maximum suppresion.
    :param roi: sorted anchors with the shape of (#anchors, 4)
    :param threshold: an real num, e.g., 0.7
    :return: the indexes of selected anchors
    '''
    rest_index = list(np.arange(0, len(roi)))
    i = 0
    while (len(rest_index) > i):
        bbox_top_score = roi[rest_index[i]]
        for index in rest_index[i+1:]:
            iou = bbox_iou(roi[index], bbox_top_score)
            if iou > threshold:
                rest_index.remove(index)
        i += 1
    return rest_index
