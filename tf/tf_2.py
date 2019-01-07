# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 10:45:35 2019

@author: dell
"""

import tensorflow as tf
import numpy as np


a = tf.convert_to_tensor(np.array([[1, 1, 2, 4], [3, 4, 8, 5]]))
print(a.dtype)
b = tf.cast(a, tf.float32)
print(b.dtype)


x = tf.constant([[1, 2], [1, 2]])
y = tf.constant([[1, 2], [1, 2]])
z = tf.add(x, y)


A = [1, 3, 4, 5, 6]
B = [1, 3, 4, 3, 2]


c = [[1., 1.], [2., 2.]]

with tf.Session() as sess:
    print(sess.run(z))
    
    print(sess.run(tf.equal(A, B)))
    print(sess.run(tf.cast(tf.equal(A, B), tf.float32)))
    
    print(sess.run(tf.reduce_mean(c)))
    print(sess.run(tf.reduce_mean(c, 0)))
    print(sess.run(tf.reduce_mean(c, 1)))
    
    print(sess.run(tf.argmax([[1, 2, 3], [4, 5, 6]], 0)))
    print(sess.run(tf.argmax([[1, 2, 3], [4, 5, 6]], 1)))
    print(sess.run(tf.argmax([[1, 0, 0]], 1)))
    print(sess.run(tf.argmax([1, 0, 0], 0)))