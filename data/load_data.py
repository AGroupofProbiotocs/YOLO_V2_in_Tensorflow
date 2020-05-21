import os
import numpy as np
from .image_processing import read_image
import xml.etree.ElementTree as ET
from utils.bbox_utils import bbox_iou
from config import FLAGS

VOC_BBOX_LABEL_NAMES = ('aeroplane',
                        'bicycle',
                        'bird',
                        'boat',
                        'bottle',
                        'bus',
                        'car',
                        'cat',
                        'chair',
                        'cow',
                        'diningtable',
                        'dog',
                        'horse',
                        'motorbike',
                        'person',
                        'pottedplant',
                        'sheep',
                        'sofa',
                        'train',
                        'tvmonitor')


def get_id_list(doc_path, file_name='trainval'):
    id_file = os.path.join(doc_path, 'ImageSets/Main/{}.txt'.format(file_name))
    id_list = [w.strip() for w in open(id_file)]
    return id_list

def get_raw_data(doc_path, img_id, use_difficult=False):
    anno = ET.parse(os.path.join(doc_path, 'Annotations', img_id + '.xml'))
    difficult = []
    bbox = []
    label = []
    for obj in anno.findall("object"):
        if not use_difficult and int(obj.find('difficult').text) == 1:
            continue

        difficult.append(int(obj.find('difficult').text))
        # subtract 1 to make pixel indexes 0-based,
        # the elements are corresponding to 'ymin', 'xmin', 'ymax', 'xmax' respectively.
        bbox.append([int(cor.text)-1 for cor in obj.find('bndbox')])
        label.append(VOC_BBOX_LABEL_NAMES.index(obj.find('name').text.lower().strip()))

    # print('Object number:', len(bbox))
    difficult = np.array(difficult, dtype=np.int32) #(#objects, )
    bbox = np.stack(bbox).astype(np.float32) #(#objects, 4)
    label = np.array(label, dtype=np.int32) #(#objects, )

    img_path = os.path.join(doc_path, 'JPEGImages/{}.jpg'.format(img_id))
    img = read_image(img_path)
    # print(img.shape)

    return img, bbox, label, difficult

def create_ground_truth(bbox, label, anchors, feat_size):
    '''
    for single image, accoording to the raw information of VOC dataset, create the ground-truth used in loss calculation.

    :param bbox: an array with the shape of (#objects, 4), make sure the bbox coordinates have been normalized to [0,1].
    :param label: an array with the shape of (#objects, )
    :param anchors: a list with 5 elements which define the anchor boxes, each element is a list like [w, h]
    :param img_size: original input image size
    :param feat_size: feature size
    :return: normalized_bbox: an array with the shape of (#objects, 4), containing the normalized ground-truth
                               coordinates of bbox, which are used to calculate the iou with pred bbox.
             matched_true_offset: an array with the shape of (feat_h, feat_w, #anchors, 4), e.g., (13, 13, 5, 4), which
                                  denotes the gt of the offset, i.e., [center_x_offset, center_x_offset, w_offset, h_offset],
                                  and it's calculated between the matched anchor and the true box and used to calculate
                                  the offset loss.
             matched_true_class_prob: an array with the shape of (feat_h, feat_w, #anchors, 20), e.g., (13, 13, 5, 20),
                                      which denotes the corresponding one-hot class label, and it's used to calculate
                                      the classification loss.
             target_anchor_mask: an array with the shape of (feat_h, feat_w, #anchors, 1), e.g., (13, 13, 5, 1), which
                                 is the mask labels that which anchor is going to used to predict the bbox, and also used
                                 as weight when calculating the yolo_v2 loss.
    '''

    num_anchors = len(anchors)
    num_objects = len(label)
    anchors = np.array(anchors, dtype='float32')
    feat_h, feat_w = feat_size

    target_anchor_mask = np.zeros((feat_h, feat_w, num_anchors, 1), dtype='float32')
    matched_true_offset = np.zeros((feat_h, feat_w, num_anchors, 4), dtype='float32')
    matched_true_class_prob = np.zeros((feat_h, feat_w, num_anchors, 20), dtype='float32')

    # calculte the width and height as well as the center coordinate of gt bboxes, and then change them into the value
    # with respect to grid.
    grid_scale = np.array([feat_w, feat_h], dtype='float32')
    bbox_wh = (bbox[:, 2:] - bbox[:, :2]) * grid_scale  # shape: (#objects, 2)
    bbox_xy = (bbox[:, :2] * grid_scale) + bbox_wh / 2  # shape: (#objects, 2)

    # find out which grid the gt bbox center belongs to, the matched grid will be selected for the prediction.
    grid_xy = bbox_xy.astype('int')

    # calculate the iou between each gt bbox and each anchor shifted to origin, i.e., only consider the shape
    bbox_coor_max = bbox_wh / 2.
    bbox_coor_min = - bbox_coor_max
    shifted_bbox = np.concatenate([bbox_coor_min, bbox_coor_max], axis=-1)   # shape: (#objects, 4)

    anchor_coor_max = anchors / 2.
    anchor_coor_min = - anchor_coor_max
    shifted_anchor = np.concatenate([anchor_coor_min, anchor_coor_max], axis=-1)  # shape: (#anchors, 4)

    iou = bbox_iou(shifted_bbox, shifted_anchor) # shape: (#objects, #anchors)

    matched_idx = np.argmax(iou, axis=1)  #It's OK if multiple bbbxes are matched to same anchor, since the grid they belong to are usually different.
    matched_anchors = anchors[matched_idx] # shape: (#objects, 2)

    target_anchor_mask[grid_xy[:, 1], grid_xy[:, 0], matched_idx, :] = 1  # use the magic indexing

    center_offset = bbox_xy - grid_xy  # shape: (#objects, 2)
    # print(bbox_wh)
    wh_offset = np.log(bbox_wh / matched_anchors)  # shape: (#objects, 2)
    gt_offset = np.concatenate([center_offset, wh_offset], axis=-1)  # shape: (#objects, 4)
    matched_true_offset[grid_xy[:, 1], grid_xy[:, 0], matched_idx, :] = gt_offset

    one_hot_label = np.zeros((num_objects, FLAGS.num_class))
    one_hot_label[np.arange(num_objects), label] = 1.
    matched_true_class_prob[grid_xy[:, 1], grid_xy[:, 0], matched_idx, :] = one_hot_label

    return matched_true_offset, matched_true_class_prob, target_anchor_mask





    












