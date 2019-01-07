# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 21:01:20 2019

@author: dell
"""
'''
前向传播过程文件
实现对网络中参数和偏置的初始化、定义卷积结构和池化结构、定义前向传播过程
'''

import tensorflow as tf

image_size = 28
num_channels = 1
conv1_size = 5
conv1_kernel_num = 32
conv2_size = 5
conv2_kernel_num = 64
fc_size = 512
output_node = 10

def get_weight(shape, regularizer): # 权重 w 生成函数
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):    # 偏置 b 生成函数
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x, w):   # 卷积层计算函数
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):    # 最大池化层计算函数
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def forward(x, train, regularizer):
    conv1_w = get_weight([conv1_size, conv1_size, num_channels, conv1_kernel_num], regularizer)
    conv1_b = get_bias([conv1_kernel_num])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)
    
    conv2_w = get_weight([conv2_size, conv2_size, conv1_kernel_num, conv2_kernel_num], regularizer)
    conv2_b = get_bias([conv2_kernel_num])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)
    
    pool_shape = pool2.get_shape().as_list() # get_shape()函数得到 pool2 的维度，并存入 list 中
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])    # pool_shape[0] 为 batch 值
    
    fc1_w = get_weight([nodes, fc_size], regularizer)
    fc1_b = get_bias([fc_size])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)
        
    fc2_w = get_weight([fc_size, output_node], regularizer)
    fc2_b = get_bias([output_node])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y