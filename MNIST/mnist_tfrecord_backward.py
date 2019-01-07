# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:33:55 2019

@author: dell
"""

"""
反向传播过程文件
反向传播过程利用训练数据集对神经网络模型训练，通过降低损失函数值，实现网络模型参数的优化
从而得到准确率高且泛化能力强的神经网络模型
"""

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os
import mnist_generateds# 1

batch_size = 200
learning_rate_base = 0.1
learning_rate_decay = 0.99
regularizer = 0.0001
Steps = 50000
moving_average_decay = 0.99
model_save_path = "./model_tfrecord/"
model_name = "mnist_model"
train_num_examples = 60000  # 2

def backward():
    x = tf.placeholder(tf.float32, [None, mnist_forward.input_node])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.output_node])
    
    # 调用前向传播过程 forward() 函数，并设置正则化，计算训练数据集上的预测结果 y
    y = mnist_forward.forward(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))  # 损失函数中加入正则化项
    
    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
            learning_rate_base,
            global_step,
            train_num_examples / batch_size,
            learning_rate_decay,
            staircase=True)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
        
    saver = tf.train.Saver()
    img_batch, label_batch = mnist_generateds.get_tfrecord(batch_size, isTrain=True)    # 3
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        coord = tf.train.Coordinator()  # 4 开启线程协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 5 启动输入队列的线程，填充训练样本到队列中
        
        for i in range(Steps):
            xs, ys = sess.run([img_batch, label_batch]) # 6
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g" % (i, loss_value))
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)
        
        coord.request_stop()    # 7 关闭线程协调器
        coord.join(threads)     # 8

def main():
    backward()  # 9
    
if __name__ == '__main__':
    main()