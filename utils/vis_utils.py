import os
import sys
import numpy as np
from PIL import Image
from data.image_processing import normalize
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
# matplotlib.use('Agg')

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

BBOX_COLORS = ( 'b',
                'g',
                'r',
                'c',
                'm',
                'y',
                'k',
                'w',
                'aliceblue',
                'aqua',
                'aquamarine',
                'azure',
                'beige',
                'bisque',
                'blanchedalmond',
                'blueviolet',
                'burlywood',
                'cadetblue',
                'chartreuse',
                'chocolate',
                'coral')

#---------------------------------------------------------------------------------------------------------------

def view_bar(step, total, extra_msg={}):
    num = step + 1
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    arrow = 0 if num==total else 1
    r = '\rStep:%d/%d [%s%s%s]%d%%' % (num, total, '■'*rate_num, '▶'*arrow, '-'*(100-rate_num-arrow), rate*100)
    for key in extra_msg.keys():
        r += ' --- %s:%f' % (key, extra_msg[key])
    sys.stdout.write(r)
    sys.stdout.flush()

def show_image(img, bbox, label, score=None, min_score_thres=None):
    if img.shape[0] == 3 or img.shape[0] == 1:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 3:
        cm = None
    elif img.shape[-1] == 1:
        img = np.squeeze(img)
        cm = 'gray'
    else:
        raise Exception('The image channel should be 1 or 3! The current channel is {}.'.format(img.shape[-1]))

    if img.dtype != np.uint8:
        img = normalize(img, [0, 1])

    plt.figure()
    plt.imshow(img, cmap=cm)
    plt.axis('off')
    currentAxis = plt.gca()

    for obj in range(len(label)):

        if (min_score_thres != None) and (score[obj] < min_score_thres):
            continue

        left = bbox[obj, 0]
        top = bbox[obj, 1]
        width = bbox[obj, 2] - bbox[obj, 0]
        height = bbox[obj, 3] - bbox[obj, 1]
        rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor=BBOX_COLORS[obj], facecolor='none')
        currentAxis.add_patch(rect)

        if score is not None:
            text = VOC_BBOX_LABEL_NAMES[int(label[obj])] + ' score: %.2f'%score[obj]
        else:
            text = VOC_BBOX_LABEL_NAMES[int(label[obj])]

        plt.text(left + 2, top - 5, text, fontsize=10,
                 bbox=dict(boxstyle="square, pad=0.2", ec=BBOX_COLORS[obj], fc='w'))
    plt.show()


def fig2array(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    return buf