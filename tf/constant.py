# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:20:25 2017

@author: dell
"""
import tensorflow as tf
import numpy as np

a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)

b = np.array([[3., 0.], 
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

print('---------------')

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])

print('---------------')

c = tf.constant(2, name="c")
d = tf.constant(3, name="d")
x = tf.add(c, d, name="add")
print(x)
with tf.Session() as sess:
   print (sess.run(x))

print('---------------')

w = 2
y = 3
add_op = tf.add(w, y)
mul_op = tf.multiply(w, y)
useless = tf.multiply(w, add_op)
pow_op = tf.pow(add_op, mul_op)
print(pow_op)
with tf.Session() as sess:
    z = sess.run(pow_op)
    print (z)

print('---------------')
    
#t1 = [[1, 2, 3], [4, 5, 6]]
#t2 = [[4, 5, 6], [1, 2, 3]]
t1 = tf.expand_dims(tf.constant([1, 2, 3], 1))
t2 = tf.expand_dims(tf.constant([4, 5, 6], 1))

concated = tf.concat([t1,t2], 1)
print(concated)