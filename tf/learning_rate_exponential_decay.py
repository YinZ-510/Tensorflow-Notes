# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 17:13:16 2019

@author: dell
"""

import tensorflow as tf

LEARNING_RATE_BASE = 0.1    # 最初学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
LEARNING_RATE_STEP = 1      # 进行多少轮 BATCH_SIZE 后，更新一次学习率
                            # 一般设为：总样本数 / BATCH_SIZE，即遍历一次全部训练数据，更新一次学习率

# 记录了当前训练轮数，初值为 0，设为不被训练
global_step = tf.Variable(0, trainable=False)

# 定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)

w = tf.Variable(tf.constant(5, dtype=tf.float32))
loss = tf.square(w+1)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 生成会话，训练 40 轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print('After %s steps: global_step is %f, w is %f, learning rate is %f, loss is %f' % 
              (i, global_step_val, w_val, learning_rate_val, loss_val))