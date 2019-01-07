# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:51:17 2019

@author: dell
"""

"""
测试过程文件
验证模型准确率
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
test_interval_secs = 20

def test(mnist):
    # 将当前图设置为默认图，并返回一个上下文管理器，用于将已经定义好的神经网络在计算图中复现
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.input_node])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.output_node])
        
        # 调用前向传播过程 forward() 函数，计算测试数据集上的预测结果 y
        y = mnist_forward.forward(x, None)
        
        # 实例化可还原滑动平均的 saver
        ema = tf.train.ExponentialMovingAverage(mnist_backward.moving_average_decay)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
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