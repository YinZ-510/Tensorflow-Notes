# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:11:32 2019

@author: dell
"""

import tensorflow as tf
import numpy as np

x = tf.constant([[1, 2, 3], [4, 5, 6]])
y = [[1, 2, 3], [4, 5, 6]]
z = np.arange(24).reshape([2, 3, 4])

sess = tf.Session()

# tf.shape(a)，a 的数据类型可以是 tensor、list、array

x_shape = tf.shape(x)
y_shape = tf.shape(y)
z_shape = tf.shape(z)

print(sess.run(x_shape))
print(sess.run(y_shape))
print(sess.run(z_shape))

'''
[2 3]
[2 3]
[2 3 4]
'''

# a.get_shape()，a 的数据类型只能是 tensor，返回元组，可以使用 as_list() 得到列表

x_shape_1 = x.get_shape()
print(x_shape_1)
x_shape_2 = x.get_shape().as_list()
print(x_shape_2)

'''
(2, 3)
[2, 3]
'''