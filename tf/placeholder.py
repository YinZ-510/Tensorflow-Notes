# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:03:39 2018

@author: dell
"""

import tensorflow as tf

'''
tf.placeholder(dtype, shape, name=None)
三个参数，dtype 表示数据类型，常用的有 tf.float32、tf.float64 等数值类型
shape 表示数据形状，name表示名称
'''

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b
with tf.Session() as sess:
    #feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
    print(sess.run(c, {a: [1, 2,3]}))