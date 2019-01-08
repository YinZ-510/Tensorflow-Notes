# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:38:58 2019

@author: dell
"""

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']    # 正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False      # 正常显示正负号

def load_image(path):
    fig = plt.figure("Centre and Resize")
    img = io.imread(path)   # 读入图片 
    img = img / 255.0       # 像素归一化到 [0, 1]
    
    ax0 = fig.add_subplot(131)  # 将画布分割成一行三列，并把下面的图像放在该画布第一个位置 
    ax0.set_xlabel(u'Original Picture') # 添加子标签
    ax0.imshow(img) 
    
    short_edge = min(img.shape[:2]) # 找到该图像的最短边 
    y = (img.shape[0] - short_edge) // 2  
    x = (img.shape[1] - short_edge) // 2 
    crop_img = img[y: y + short_edge, x: x + short_edge] 
    
    ax1 = fig.add_subplot(132)  # 把下面的图像放在该画布第二个位置
    ax1.set_xlabel(u"Centre Picture")   # 添加子标签
    ax1.imshow(crop_img)
    
    re_img = transform.resize(crop_img, (224, 224))     # resize 成固定的 img_size
    
    ax2 = fig.add_subplot(133)  # 把下面的图像放在该画布第三个位置
    ax2.set_xlabel(u"Resize Picture")   # 添加子标签
    ax2.imshow(re_img)
	
    img_ready = re_img.reshape((1, 224, 224, 3))

    return img_ready

# 定义百分比转换函数
def percent(value):
    return '%.2f%%' % (value * 100)