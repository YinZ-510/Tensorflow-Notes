# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:43:10 2019

@author: dell
"""

"""
测试过程文件
验证模型准确率
"""

import time
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_tfrecord_backward
import mnist_generateds
test_interval_secs = 20
test_num = 10000    # 1

def test():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.input_node])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.output_node])
        
        # 调用前向传播过程 forward() 函数，计算测试数据集上的预测结果 y
        y = mnist_forward.forward(x, None)
        
        # 实例化可还原滑动平均的 saver
        ema = tf.train.ExponentialMovingAverage(mnist_tfrecord_backward.moving_average_decay)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        img_batch, label_batch = mnist_generateds.get_tfrecord(test_num, isTrain=False)  # 2
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_tfrecord_backward.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    
                    coord = tf.train.Coordinator()  # 3开启线程协调器
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 4 启动输入队列的线程，填充测试样本到队列中
                    
                    xs, ys = sess.run([img_batch, label_batch]) # 5
                    
                    accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                    
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    
                    coord.request_stop()    # 6 关闭线程协调器
                    coord.join(threads)     # 7
                    
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(test_interval_secs)
            
def main():
    test()  # 8
    
if __name__ == '__main__':
    main()