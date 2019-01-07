# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 13:59:00 2019

@author: dell
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_forward
import mnist_backward
import mnist_tfrecord_backward
import os

def restore_model(testPicArr):
    # 创建一个默认图，在该图中执行以下操作（多数操作和 train 中一样）
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.input_node])
        # 调用前向传播过程 forward() 函数，计算预测结果 y
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)  # 得到概率最大的预测值
        
        # 实现滑动平均模型，参数 moving_average_decay 用于控制模型更新的速度
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        with tf.Session() as sess:
            # 通过 checkpoint 文件定位到最新保存的模型
            ckpt = tf.train.get_checkpoint_state(mnist_backward.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1
           
# 预处理函数，包括 resize、转变灰度图、二值化操作
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50  # 设定合理的阈值
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
                
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)
    
    return img_ready

def application():
    img_dir = "pic/"
    img_list = os.listdir(img_dir)
    for i in range(len(img_list)):
        pre_img = os.path.join(img_dir, img_list[i])
        pre_img = pre_pic(pre_img)
        preValue = restore_model(pre_img)
        print("%s: %d" % (img_list[i], preValue))
    
#    testNum = input("input the number of test pictures:")
#    for i in range(int(testNum)):
#        testPic = input("the path of test picture:")
#        testPicArr = pre_pic(testPic)
#        preValue = restore_model(testPicArr)
#        print("The prediction number is:", preValue)
        
def main():
    application()
    
if __name__ == '__main__':
    main()