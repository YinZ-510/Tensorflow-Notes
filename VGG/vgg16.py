# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 14:36:57 2019

@author: dell
"""

import inspect
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

vgg_mean = [103.939, 116.779, 123.68]   # 样本 RGB 均值

class Vgg16():
    
    def __init__(self, vgg16_path=None):
        if vgg16_path is None:
            vgg16_path = os.path.join(os.getcwd(), "vgg16.npy")
            print(vgg16_path)
            self.data_dict = np.load(vgg16_path, encoding='latin1').item()  # 遍历键值对，导入模型参数
            
        for x in self.data_dict:
            print(x)
            
    def forward(self, images):
        print("build model started")
        start_time = time.time()
        rgb_scaled = images * 255.0 # 逐像素乘以 255.0
        red, green, blue = tf.split(rgb_scaled, 3, 3)
        # 从 RGB 转换为 BGR
        assert red.get_shape().as_list()[1:] == [224, 224, 1]   # 判断每个操作后的维度变化是否和预期一致
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        
        bgr = tf.concat([
                blue - vgg_mean[0],
                green - vgg_mean[1],
                red - vgg_mean[2]], 3)  # 逐样本减去每个通道的像素平均值，可以移除图像的平均亮度值
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        
        # 构建 VGG 16 层网络，逐层根据传入命名空间的 name 读取对应层的网络参数
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        # 传入 conv1_2 名字，获取该层的卷积核和偏置，进行卷积运算，返回经过激活函数后的值
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        # 传入 pool1 名字，对该层进行相应的池化操作
        self.pool1 = self.max_pool_2x2(self.conv1_2, "pool1")
        
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool_2x2(self.conv2_2, "pool2")
        
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool_2x2(self.conv3_3, "pool3")
        
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool_2x2(self.conv4_3, "pool4")
        
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool_2x2(self.conv5_3, "pool5")
        
        self.fc6 = self.fc_layer(self.pool5, "fc6") # 根据命名空间的 name 进行加权求和运算
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)
        
        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        
        self.fc8 = self.fc_layer(self.relu7, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")
        
        end_time = time.time()
        print("time consuming: %f" % (end_time - start_time))
        
        self.data_dict = None   # 清空本次读取到的模型参数字典
        
    # 定义卷积运算
    def conv_layer(self, x, name):
        with tf.variable_scope(name):   # 根据命名空间的 name 找到对应卷积层的网络参数
            w = self.get_conv_filter(name)  # 获取该层的卷积核
            conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME') # 卷积计算
            conv_biases = self.get_bias(name)   # 获取该层偏置项
            result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))  # 加上偏置，并做激活计算
            return result
    
    # 定义获取卷积核的函数
    def get_conv_filter(self, name):
        # 根据命名空间的 name 从参数字典获取对应层的卷积核
        return tf.constant(self.data_dict[name][0], name="filter")
    
    # 定义获取偏置项的函数
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")
    
    # 定义最大池化操作
    def max_pool_2x2(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    # 定义全连接层的前向传播计算
    def fc_layer(self, x, name):
        with tf.variable_scope(name):    # 根据命名空间的 name 做全连接层计算
            shape = x.get_shape().as_list()
            print("fc_layer shape:", shape)
            dim = 1
            for i in shape[1:]:
                dim *= i    # 将每层维度相乘
            x = tf.reshape(x, [-1, dim])    # 改变特征图的形状
            
            w = self.get_fc_weight(name)    # 获取权重值
            b = self.get_bias(name)         # 获取偏置项值
            result = tf.nn.bias_add(tf.matmul(x, w), b)
            return result
    
    # 定义获取权重的函数
    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
