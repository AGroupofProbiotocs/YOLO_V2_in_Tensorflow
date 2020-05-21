from numpy import random
import os
import numpy as np
from .load_data import get_raw_data, create_ground_truth
from .image_processing import *
import cv2
from PIL import Image
from imgaug import augmenters as iaa
import imgaug as ia
from config import FLAGS

chance = FLAGS.augment_chance
seq = iaa.Sequential([
        # change brightness of images (by -10 to 10 of original value)
        iaa.Sometimes(chance, iaa.Add((-FLAGS.intensity_change, FLAGS.intensity_change), per_channel=0.5)),
        # change brightness of images (85-115% of original value)
        iaa.Sometimes(chance, iaa.Multiply((1-FLAGS.brightness_change, 1+FLAGS.brightness_change), per_channel=0.5)),
        # improve or worsen the contrast
        iaa.Sometimes(chance, iaa.ContrastNormalization((1-FLAGS.contrast_change, 1+FLAGS.contrast_change), per_channel=0.5)),
        ],
    random_order=True # do all of the above in random order
)

def generate_train_batch(doc_path, id_list, anchors, batch_size, img_size, feat_size, random_shuffle=True, preprocess=False,
                         augment=False, save_to_dir=None):
    '''
    A generator that yields a batch of (img, bbox, label). The batch size is 1.
    :param doc_path: the image document path
    :param id_list: the image id list
    :param anchors: a list with 5 elements which define the anchor boxes, each element is a list like [w, h]
    :param batch_size: batch size
    :param image size: image size
    :param feature_size: the last feature map size
    :param random_shuffle: whether to shuffle the images
    :param preprocess: whether to preprocess the images
    :param augment: whether to augment the images
    :param save_to_dir: the document direction to save the images
    :return: normalized_bboxes: (batch_size, max_#object, 4)
             matched_true_offsets : (batch_size, feat_h, feat_w, #anchors, 4)
             matched_true_class_probs : (batch_size, feat_h, feat_w, #anchors, 20)
             target_anchor_masks : (batch_size, feat_h, feat_w, #anchors, 1)
    '''

    N = len(id_list)
    num_anchors = len(anchors)
    feat_h, feat_w = feat_size
    img_h, img_w = img_size

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        # if reach last batch, tune the size of batch
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        input_images = np.zeros((current_batch_size, img_h, img_w, 3), dtype='float32')
        bboxes = [] #the bboxes will stored in a list first, since the max #objects are decided at last
        matched_true_offsets = np.zeros((current_batch_size, feat_h, feat_w, num_anchors, 4), dtype='float32')
        matched_true_class_probs = np.zeros((current_batch_size, feat_h, feat_w, num_anchors, 20), dtype='float32')
        target_anchor_masks = np.zeros((current_batch_size, feat_h, feat_w, num_anchors, 1), dtype='float32')

        max_obj_num = 0

        for i in range(current_index, current_index + current_batch_size):

            img, bbox, label, _ = get_raw_data(doc_path, id_list[i])

            # to find the max object number
            obj_num = bbox.shape[0]
            if obj_num > max_obj_num:
                max_obj_num = obj_num

            if augment:
                img, bbox, label = random_crop(img, bbox, label)
                img, bbox = random_fliplr(img, bbox)
                img = seq.augment_image(img)

            if preprocess:
                img, bbox = data_preprocess(img, bbox)  # the bbox will be normalized to [0,1]

            if save_to_dir:
                Img = Image.fromarray(np.uint8(img))
                file_end = random.randint(0, 100000)
                while os.path.exists(os.path.join(save_to_dir, id_list[i], str(file_end))) is not True:
                    file_end = random.randint(0, 100000)
                Img.save(os.path.join(save_to_dir, id_list[i], str(file_end)))

            true_offset, true_prob, target_mask = create_ground_truth(bbox, label, anchors, feat_size)

            # fill the batch
            input_images[i - current_index, ...] = img
            bboxes.append(bbox)
            matched_true_offsets[i - current_index, ...] = true_offset
            matched_true_class_probs[i - current_index, ...] = true_prob
            target_anchor_masks[i - current_index, ...] = target_mask

        # define the normalized_bboxes array and fill it
        normalized_bboxes = np.zeros((current_batch_size, max_obj_num, 4), dtype='float32')
        for j, cur_bbox in enumerate(bboxes):
            cur_obj_num = cur_bbox.shape[0]
            normalized_bboxes[j, :cur_obj_num, :] = cur_bbox

        # 'batch_index == 0' means a new round of iteration
        if random_shuffle and batch_index == 0:
            random.shuffle(id_list)

        yield (input_images, normalized_bboxes, matched_true_offsets, matched_true_class_probs, target_anchor_masks)


def generate_val_batch(doc_path, id_list, random_shuffle=True, preprocess=False, augment=False, save_to_dir=None,
                       use_difficult=False):
    '''
    A generator that yields a batch of (img, bbox, label) for validation. The batch size is fixed to 1.
    '''

    count = 0
    N = len(id_list)

    while True:
        img, bbox, label, diffucult = get_raw_data(doc_path, id_list[count], use_difficult=use_difficult)

        if augment:
            img, bbox, label = random_crop(img, bbox, label)
            img, bbox = random_fliplr(img, bbox)
            img = seq.augment_image(img)

        if preprocess:
            img, bbox = data_preprocess(img, bbox, bbox_normalize=False)

        if save_to_dir:
            Img = Image.fromarray(np.uint8(img))
            file_end = random.randint(0, 100000)
            while os.path.exists(os.path.join(save_to_dir, id_list[count], str(file_end))) is not True:
                file_end = random.randint(0, 100000)
            Img.save(os.path.join(save_to_dir, id_list[count], str(file_end)))

        if count == N-1:
            count = 0
            if random_shuffle:
                random.shuffle(id_list)
        else:
            count += 1

        # add a new dimension so that it can be fed into the network
        img = img[None, ...]

        if use_difficult:
            yield (img, bbox, label, diffucult)
        else:
            yield (img, bbox, label)

