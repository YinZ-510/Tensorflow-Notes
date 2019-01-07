# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:10:34 2019

@author: dell
"""

'''
测试过程文件
对 mnist 数据集中的测试数据进行预测，测试模型准确率
'''

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import mnist_lenet5_backward
import numpy as np

test_interval_secs = 10

def test(mnist):
    # 将当前图设置为默认图，并返回一个上下文管理器，用于将已经定义好的神经网络在计算图中复现
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[
                mnist.test.num_examples,
                mnist_lenet5_forward.image_size,
                mnist_lenet5_forward.image_size,
                mnist_lenet5_forward.num_channels])
        y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.output_node])
        y = mnist_lenet5_forward.forward(x, False, None)
        
        ema = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.moving_average_decay)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_x = np.reshape(mnist.test.images, (
                            mnist.test.num_examples,
                            mnist_lenet5_forward.image_size,
                            mnist_lenet5_forward.image_size,
                            mnist_lenet5_forward.num_channels))
                    accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_x, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(test_interval_secs)
        
def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)
    
if __name__ == '__main__':
    main()