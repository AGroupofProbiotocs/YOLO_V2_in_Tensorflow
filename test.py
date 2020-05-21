import numpy as np
import tensorflow as tf
import os
from model.darknet import DarkNet448
from utils.bbox_utils import bbox_iou
import time

darknet_cfg_path = './pre_weight/darknet19_448.cfg'

def create_cfg_dict():
    cfg = {}
    count = 0
    section_name = ''
    lines = open(darknet_cfg_path).readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('['):
            section_name = line.strip('[]') + '_' + str(count)
            cfg[section_name] = {}
            count += 1
        elif line == '':
            continue
        else:
            if section_name == '':
                raise LookupError('The file format is unsupproted!')
            else:
                para_name, para_value = line.split('=')
                cfg[section_name][para_name.strip()] = para_value.strip()
    return cfg

# d = create_cfg_dict()
# print(d)


# # test the bn initialization
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#
# x = np.random.randn(1,3,3,3).astype('float32')
# print(x.shape[-1:])
#
# gamma = np.array([1,2,3]).astype('float32')
# beta = np.array([0,0,0]).astype('float32')
# mean = np.array([0,0,0]).astype('float32')
# var = np.array([1,1,1]).astype('float32')
#
# y1 = gamma*((x-mean)/np.sqrt(var+0.001))+beta
# print(y1)
#
# y3 = tf.nn.batch_normalization(x, mean, var, beta, gamma, 0.001)
#
# gamma_init = tf.constant_initializer(gamma)
# beta_init = tf.constant_initializer(beta)
# mean_init = tf.constant_initializer(mean)
# var_init = tf.constant_initializer(var)
# y2 = tf.contrib.layers.batch_norm(x, center = True, scale = True, is_training = False,
#                                  param_initializers = {'gamma':gamma_init, 'beta':beta_init, 'moving_mean':mean_init, 'moving_variance':var_init})
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     y2 = sess.run(y2)
#     print(y2)
#     # print(y2 == x)
#     y3 = sess.run(y3)
#     print(y3)

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# a = DarkNet448()
# a.save_ckpt()
# print(isinstance(a, DarkNet448))

# a=[0,1,2,3]
# b=[0,1,2,3,4]
# c=tf.meshgrid(b,a)
# e = tf.expand_dims(c[0], axis=0)

a = tf.ones((2,3))
b = tf.cast(a>0, dtype=tf.float32)
with tf.Session() as sess:
    print(sess.run(b))
    # d = sess.run(c)
    # print(d[0])
    # print(d[1])
    # print(sess.run(e).shape)

# a = np.array([[-1.2,-2.1, 1.2, 2.1], [-4.2,-3.1, 4.2, 3.1]])
# b = np.array([[-1.4,-2.3, 1.4, 2.3], [-4.1,-3.2, 4.1, 3.2], [-4.0,-3.3, 4.0, 3.3]])
# start = time.time()
# iou = bbox_iou(a,b)
# idx = np.argmax(iou, axis=1)
# end = time.time()
# print(iou)
# print(idx)
# print(end-start)