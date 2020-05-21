import os
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import random
import matplotlib
from config import FLAGS
matplotlib.use('Agg')


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

def normalize(data, interval):
    '''
    [lower, upper] = interval
    upper: maximum value
    lower: minimum value 
    '''    
    [lower, upper] = interval
    mx = np.max(data)
    mn = np.min(data)
    if mx==mn:
#        print('大小', data.shape)
#        plt.imshow(data[5], cmap='gray')
        norm_data = np.zeros(data.shape)
    else:  
        norm_data = (upper-lower)*(data - mn) / (mx - mn) + lower
    return norm_data.astype(np.float32)

def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is HWC format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image with shape of (h, w, c)
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.array(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        return img[..., np.newaxis]
    else:
        return img

def minmaxmap(data, lower, upper):
    mx = np.max(data)
    mn = np.min(data)
    if mx==mn:
#        print('大小', data.shape)
#        plt.imshow(data[5], cmap='gray')
        norm_data = np.zeros(data.shape)
    else:
        norm_data = (upper-lower)*(data - mn) / (mx - mn) + lower
    return norm_data

def data_preprocess(img, bbox, size=[448, 448], resize=True, img_normalize=True, bbox_normalize=True):
    org_h, org_w, _ = img.shape
    if resize:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    if img_normalize:
        img = img / 255.
    if bbox_normalize:
        # Normalize the bbox coordinate to [0,1] according to the input image size
        norm_scale = np.array([org_w, org_h, org_w, org_h], dtype='float32')
        bbox = bbox / norm_scale

    return img, bbox

def bbox_rescale(bbox, scale):
    out_bbox = bbox*scale
    return out_bbox

def random_crop(img, bbox, label, x_range=[0.01, 0.1], y_range=[0.01, 0.1], min_edge = FLAGS.min_edge):
    '''
    :param img: with shape of (h, w, 3)
    :param bbox: with shape of (#objects, 4)
    :param label: with shape of (#objects, )
    :return:
    '''
    H, W, C = img.shape
    x_scale = random.uniform(x_range[0], x_range[1])
    y_scale = random.uniform(y_range[0], y_range[1])
    x_crop = int(W * x_scale)
    y_crop = int(H * y_scale)

    x_crop_head = random.randint(0, x_crop)
    x_crop_tail = W - (x_crop - x_crop_head)
    y_crop_head = random.randint(0, y_crop)
    y_crop_tail = H - (y_crop - y_crop_head)

    cropped_img = img[y_crop_head:y_crop_tail, x_crop_head:x_crop_tail]

    invalid_obj = []
    for obj in range(bbox.shape[0]):
        left = bbox[obj, 0]
        top = bbox[obj, 1]
        right = bbox[obj, 2]
        bottom = bbox[obj, 3]

        if left <= x_crop_head:
            bbox[obj, 0] = 0
        else:
            bbox[obj, 0] = left - x_crop_head
        if top <= y_crop_head:
            bbox[obj, 1] = 0
        else:
            bbox[obj, 1] = top - y_crop_head
        if right >= x_crop_tail:
            bbox[obj, 2] = x_crop_tail - 1 - x_crop_head
        else:
            bbox[obj, 2] = right - x_crop_head
        if bottom >= y_crop_tail:
            bbox[obj, 3] = y_crop_tail - 1 - y_crop_head
        else:
            bbox[obj, 3] = bottom - y_crop_head

        #if all the edges of a bbox are out of the image, consider it as invalid bbox and record it
        if (bbox[obj, 3] - bbox[obj, 1] < min_edge) or (bbox[obj, 2] - bbox[obj, 0] < min_edge):
            invalid_obj.append(obj)

    if invalid_obj:
        #delete the invalid bboxes
        bbox = np.delete(bbox, invalid_obj, axis=0)
        # at the same time, delete the invalid label
        label = np.delete(label, invalid_obj)

    return cropped_img, bbox, label

def random_fliplr(img, bbox):
    H, W, C = img.shape
    flip = random.choice([True, False])
    if flip:
        img = np.fliplr(img)
        bbox[:, [0, 2]] = (W - 1) - bbox[:, [2, 0]]

    return img, bbox






