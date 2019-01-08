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
'''
Tensor:
 a = Tensor("Const:0", shape=(2, 2), dtype=float32)
NumpyArray:
 b = [[ 3.  0.]
 [ 5.  1.]]
'''

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])
'''
Tensor("strided_slice_1:0", shape=(), dtype=float32)
Tensor("strided_slice_3:0", shape=(), dtype=float32)
Tensor("strided_slice_5:0", shape=(), dtype=float32)
Tensor("strided_slice_7:0", shape=(), dtype=float32)
'''

c = tf.constant(2, name="c")
d = tf.constant(3, name="d")
x = tf.add(c, d, name="add")
print(x)
with tf.Session() as sess:
   print(sess.run(x))
'''
Tensor("add:0", shape=(), dtype=int32)
5
'''

w = 2
y = 3
add_op = tf.add(w, y)
mul_op = tf.multiply(w, y)
useless = tf.multiply(w, add_op)
pow_op = tf.pow(add_op, mul_op)
print(pow_op)
with tf.Session() as sess:
    z = sess.run(pow_op)
    print(z)
'''
Tensor("Pow:0", shape=(), dtype=int32)
15625
'''  

t1 = tf.constant([1, 2, 3])
t2 = tf.constant([4, 5, 6])
print(t1)
print(t2)
'''
Tensor("Const_1:0", shape=(3,), dtype=int32)
Tensor("Const_2:0", shape=(3,), dtype=int32)
'''

t1 = tf.expand_dims(t1, 1)
t2 = tf.expand_dims(t2, 1)
print(t1)
print(t2)
'''
Tensor("ExpandDims:0", shape=(3, 1), dtype=int32)
[[1]
 [2]
 [3]]
Tensor("ExpandDims_1:0", shape=(3, 1), dtype=int32)
[[4]
 [5]
 [6]]
'''

# tf.concat()，axis = 0，第 0 维度连接 tensor，即增加行数；axis = 1，第 1 维度连接 tensor，即增加列数
print(tf.concat([t1,t2], 0))
'''
Tensor("concat:0", shape=(6, 1), dtype=int32)
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]]
'''

print(tf.concat([t1,t2], 1))
'''
Tensor("concat_1:0", shape=(3, 2), dtype=int32)
[[1 4]
 [2 5]
 [3 6]]
'''