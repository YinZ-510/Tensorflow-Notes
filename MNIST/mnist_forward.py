# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:21:39 2019

@author: dell
"""

"""
描述网络结构的前向传播文件
在前向传播过程中，需要定义网络模型输入层个数、隐藏层节点数、输出层个数，定义网络参数 w、偏置 b
定义由输入到输出的神经网络架构
"""

import tensorflow as tf

input_node = 784
output_node = 10
layer1_node = 500

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: # 正则化
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, regularizer):
    w1 = get_weight([input_node, layer1_node], regularizer)
    b1 = get_bias([layer1_node])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    
    w2 = get_weight([layer1_node, output_node], regularizer)
    b2 = get_bias([output_node])
    y = tf.matmul(y1, w2) + b2
    return y