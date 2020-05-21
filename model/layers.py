import tensorflow as tf
import numpy as np


def glb_avg_pool(x, name):
    return tf.reduce_mean(x, axis=[1, 2], keep_dims=True, name=name)


def avg_pool(x, kernel_size, stride, name):
    return tf.nn.avg_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1], padding='SAME',name=name)


def max_pool(x, kernel_size, stride, name):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1], padding='SAME',name=name)


def conv2d_layer(x,
               kernel_size,
               out_channels,
               stride=1,
               padding='SAME',
               activation_func=tf.nn.leaky_relu,
               pre_weight=None,
               use_bias=False,
               pre_bias=None,
               batch_norm=True,
               bn_initializers=None,
               regularizer=None,
               name='conv',
               reuse=False,
               trainable=True,
               is_training=False):
    with tf.variable_scope(name, reuse=reuse):
        in_channels = x.get_shape()[-1].value
        filter = __get_conv_filter(kernel_size, in_channels, out_channels, pre_weight,
                                        regularizer=regularizer, trainable=trainable)

        output = tf.nn.conv2d(x, filter, [1, stride, stride, 1], padding=padding)

        if use_bias:
            conv_biases = __get_bias(out_channels, pre_bias, trainable=trainable)
            output = tf.nn.bias_add(output, conv_biases)

        if batch_norm:
            output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training,
                                                  param_initializers=bn_initializers)

        if activation_func is not None:
            output = activation_func(output)

        return output


def dense_layer(x,
                out_channels,
                activation_func=tf.nn.leaky_relu,
                pre_weight=None,
                use_bias=False,
                pre_bias=None,
                batch_norm=True,
                bn_initializers=None,
                regularizer=None,
                name='dense',
                reuse=False,
                trainable=True,
                is_training=False):
    with tf.variable_scope(name, reuse=reuse):
        shape = x.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])
        in_channels = dim
        weights = __get_dense_weight(in_channels, out_channels, pre_weight,
                                          regularizer=regularizer, trainable=trainable)

        output = tf.matmul(x, weights)

        if use_bias:
            conv_biases = __get_bias(out_channels, pre_bias, trainable=trainable)
            output = tf.nn.bias_add(output, conv_biases)

        if batch_norm:
            output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training,
                                                  param_initializers=bn_initializers)

        if activation_func is not None:
            output = activation_func(output)

        return output


def __get_conv_filter(kernel_size, in_channels, out_channels, pre_weight, regularizer, trainable):
    '''
    Create the filter/kernel variable for convolutional.
    :param kernel_size:
    :param in_channels:
    :param out_channels:
    :param pre_weight:
    :param trainable:
    :return:
    '''
    filter_shape = [kernel_size, kernel_size, in_channels, out_channels]
    if pre_weight is not None:
        init = tf.constant_initializer(pre_weight)
    else:
        init = tf.contrib.layers.variance_scaling_initializer()  # He_init
    return tf.get_variable('conv_weights', filter_shape, initializer=init, regularizer=regularizer, trainable=trainable)


def __get_bias(out_channels, pre_weight, trainable):
    '''
    Create the bias variable for convolutional.
    :param out_channels:
    :param pre_weight:
    :param trainable:
    :return:
    '''
    if pre_weight is not None:
        init = tf.constant_initializer(pre_weight)
    else:
        init = tf.truncated_normal_initializer(0., 0.001)
    return tf.get_variable('biases', [out_channels], initializer=init, trainable=trainable)


def __get_dense_weight(in_channels, out_channels, pre_weight, regularizer, trainable):
    '''
    Create the weight variable for fully connection.
    :param in_channels:
    :param out_channels:
    :param pre_weight:
    :param trainable:
    :return:
    '''
    filter_shape = [in_channels, out_channels]
    if pre_weight is not None:
        init = tf.constant_initializer(pre_weight)
    else:
        init = tf.contrib.layers.variance_scaling_initializer()  # He_init
    return tf.get_variable('dense_weights', filter_shape, initializer=init, regularizer=regularizer,
                           trainable=trainable)