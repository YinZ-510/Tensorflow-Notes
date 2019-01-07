# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 14:49:20 2019

@author: dell
"""

import tensorflow as tf
import numpy as np

batch_size = 8
seed = 23455

# 基于 seed 产生随机数
rng = np.random.RandomState(seed)
# 随机返回 32 行 2 列的矩阵，作为输入
X = rng.rand(32, 2)
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print('X:\n', X)
print('Y:\n', Y)


# 定义神经网络的输入、参数和输出，定义前向传播过程

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


# 定义损失函数及反向传播方法

loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentOptimizer(0.001, 0.9).minimize(loss)
#train_step = tf.train.Adamoptimizer(0.001).minimize(loss)


# 生成会话，训练 Steps 轮

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))
    print('\n')
    
    STEPS = 3000
    for i in range(STEPS):
        start = (i * batch_size) % 32
        end = start + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('After %d training step(s), loss on all data is %g' % (i, total_loss))
    
    print('\n')        
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))