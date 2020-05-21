'''
obtained form machrisaa's github project:
'''
import inspect
import os
import numpy as np
import tensorflow as tf
import time
from model.layers import *

class DarkNet448:
    def __init__(self, mode='pretrain', darknet_cfg_path=None, darknet_weight_path=None):
        if mode == 'pretrain':
            if darknet_cfg_path == None:
                self.darknet_cfg_path = './pre_weight/raw_weight/darknet19_448.cfg'
            else:
                self.darknet_cfg_path = darknet_cfg_path

            if darknet_weight_path is None:
                self.darknet_weight_path = './pre_weight/raw_weight/darknet19_448.weights'
            else:
                self.darknet_weight_path = darknet_weight_path
        else:
            self.darknet_cfg_path = darknet_cfg_path
            self.darknet_weight_path = darknet_cfg_path

        self.all_layers = {}

    def create_cfg_dict(self):
        '''
        Convert the .cfg file into a dictionary.
        :return: a config dictionary.
        '''
        cfg = {}
        count = 0
        section_name = ''
        lines = open(self.darknet_cfg_path).readlines()
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
                    raise LookupError('Unsupproted file format!')
                else:
                    para_name, para_value = line.split('=')
                    cfg[section_name][para_name.strip()] = para_value.strip()
        return cfg

    def forward(self, input, cfg_dict=None, reuse=None, is_training=False, name='darknet448'):
        # create the config dictionary
        if cfg_dict is None:
            cfg_dict = self.create_cfg_dict()
        # load the pre-trained weight
        weights_file = open(self.darknet_weight_path, 'rb')
        weights_header = np.frombuffer(weights_file.read(16))
        print('Weights Header: ', weights_header)

        image_height = int(cfg_dict['net_0']['height'])
        image_width = int(cfg_dict['net_0']['width'])
        weight_decay = float(cfg_dict['net_0']['decay'])
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        
        print('Building graph of darknet...')
        with tf.variable_scope(name, reuse=reuse):
            prev_layer = input
            self.all_layers['input'] = input
            for section, paras in cfg_dict.items():
                print('Current Section: ', section)

                if section.startswith('convolutional'):
                    filters = int(paras['filters'])
                    kernel_size = int(paras['size'])
                    stride = int(paras['stride'])
                    pad = int(paras['pad'])
                    activation = paras['activation']
                    batch_normalize = 'batch_normalize' in paras   # Ture or False
    
                    # padding='same' is equivalent to Darknet pad=1
                    if pad == 1:
                        padding = 'SAME'
                    else:
                        padding = 'VALID'

                    # handle the activation config
                    if activation=='linear':
                        act_func = None
                    elif activation=='leaky':
                        act_func = tf.nn.leaky_relu
                    else:
                        raise ValueError('Unsupported acativation function!')

                    # load the filter weight and bias. Note that the pretrained weights are stored as float32 format,
                    # thus each number occupies 4 bytes. The data in weight_file is arranged in the order "bias -> bn_paras -> weights".
                    conv_bias = np.frombuffer(weights_file.read(filters * 4), dtype='float32').reshape((filters,)) #bias

                    if batch_normalize:
                        use_bias = False
                        bn_weights = np.frombuffer(weights_file.read(filters * 3 * 4), dtype='float32').reshape((3, filters)) #bn paras
                        gamma_init = tf.constant_initializer(bn_weights[0])  # scale gamma
                        beta_init = tf.constant_initializer(conv_bias)  # offset beta
                        mean_init = tf.constant_initializer(bn_weights[1])  # moving mean
                        var_init = tf.constant_initializer(bn_weights[2])  # moving var
                        bn_initializers = {'gamma':gamma_init, 'beta':beta_init, 'moving_mean':mean_init, 'moving_variance':var_init}
                    else:
                        use_bias = True

                    in_dim = prev_layer.get_shape().as_list()[-1]
                    weight_size = kernel_size * kernel_size * in_dim * filters  #calculate the number of weight to get the bytes length
                    weight_shape = (filters, in_dim, kernel_size, kernel_size)  #pre-trained weight are stored in caffe style, i.e., (out_dim, in_dim, size, size)
                    conv_weight = np.frombuffer(weights_file.read(weight_size * 4), dtype='float32').reshape(weight_shape)  # weight
                    conv_weight = np.transpose(conv_weight, [2, 3, 1, 0])  #convert to tensorflow style, i.e., (size, size, in_dim, out_dim)

                    # build the graph
                    output = conv2d_layer(prev_layer, kernel_size, filters, stride, padding=padding,
                                          pre_weight=conv_weight, use_bias=use_bias, pre_bias=conv_bias,
                                          batch_norm=batch_normalize, bn_initializers=bn_initializers,
                                          regularizer=regularizer, activation_func=act_func, name=section, is_training=is_training)

                    # update the prev_layer for the use of next layer
                    prev_layer = output
                    # add the network node into the all_layers dictionary
                    self.all_layers[section] = output

                elif section.startswith('maxpool'):
                    kernel_size = int(paras['size'])
                    stride = int(paras['stride'])
                    output = max_pool(prev_layer, kernel_size, stride, name=section)

                    prev_layer = output
                    self.all_layers[section] = output

                elif section.startswith('avgpool'):
                    if paras:
                        print("Only globale average pooling is used in DarkNet.")
                    output = glb_avg_pool(prev_layer, name=section)

                    prev_layer = output
                    self.all_layers[section] = output

                elif section.startswith('route'):
                    ids = [int(i) for i in paras['layers'].split(',')]
                    layers = [self.all_layers[list(self.all_layers.keys())[i]] for i in ids]
                    if len(layers) > 1:
                        output = tf.concat(layers, axis=-1, name=section)
                    else:
                        output = layers[0]  # only one layer to route

                    prev_layer = output
                    self.all_layers[section] = output

                elif section.startswith('reorg'):
                    block_size = int(paras['stride'])
                    if block_size != 2:
                        raise ValueError('Only reorg with stride 2 is supported.')
                    output = tf.space_to_depth(prev_layer, 2, name=section)

                    prev_layer = output
                    self.all_layers[section] = output

                elif section.startswith('softmax'):
                    prev_layer = tf.contrib.layers.flatten(prev_layer)
                    output = tf.nn.softmax(prev_layer, name=section)

                    prev_layer = output
                    self.all_layers[section] = output

                elif (section.startswith('region') or section.startswith('cost') or section.startswith('net')):
                    pass

                else:
                    raise ValueError('Unsupported section header type: {}'.format(section))

        return self.all_layers[list(self.all_layers.keys())[-1]]

    def save_ckpt(self, ckpt_path = './pre_weight/tensorflow_ckpt/darknet_448.ckpt'):
        input = tf.placeholder(tf.float32, [None, None, None, 3], name='input')
        _ = self.forward(input)
        print('Graph built successfully!')

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.save(sess, ckpt_path)
            print('Ckpt saved successfully!')