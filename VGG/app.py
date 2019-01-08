# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:44:50 2019

@author: dell
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import vgg16
import utils
from Nclasses import labels


img_path = input('Input the path and image name:')
img_ready = utils.load_image(img_path) 

fig = plt.figure(u"Top-5 预测结果")     # 定义一个 figure 窗口 

with tf.Session() as sess:
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg = vgg16.Vgg16()     # 类 Vgg16 实例化为 vgg
    vgg.forward(images)     # 调用类的方法 forward()，传入待测试图像
    probability = sess.run(vgg.prob, feed_dict={images: img_ready})
    top5 = np.argsort(probability[0])[-1:-6:-1]     # 取出预测概率最大的五个索引值
    print("top5:", top5)
    values = []     # 预测概率值
    bar_label = []  # 标签值
    for n, i in enumerate(top5): 
        print("n:",n)
        print("i:",i)
        values.append(probability[0][i]) 
        bar_label.append(labels[i]) 
        print(i, ":", labels[i], "----", utils.percent(probability[0][i])) 
        
    ax = fig.add_subplot(111)   # 将会不划分为一行一列，并把下面的图像放入
    ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g') # 绘制柱状图
    ax.set_ylabel(u'probabilityit')     # 设置横轴标签
    ax.set_title(u'Top-5')      # 添加标题
    for a,b in zip(range(len(values)), values):
        # 在每个柱子的顶端添加对应的预测概率值
        ax.text(a, b+0.0005, utils.percent(b), ha='center', va = 'bottom', fontsize=7)
    plt.savefig('result.jpg')
    plt.show() 