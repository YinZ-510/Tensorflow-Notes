# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:46:59 2019

@author: dell
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os


image_train_path = './mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path = './mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train = './tfrecord_data/mnist_train.tfrecords'
image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test = './tfrecord_data/mnist_test.tfrecords'
data_path = './tfrecord_data'
resize_height = 28
resize_width = 28


# 生成 tfrecords 文件
def write_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)  # 新建一个 writer
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close
    for content in contents:
        value = content.split()
        img_path = image_path + value[0]
        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1
        
        example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                }))     # 把每张图片和标签封装到 example 中
        writer.write(example.SerializeToString())   # 把 example 序列化成字符串存储
        num_pic += 1
        print("the number of picture: ", num_pic)
    writer.close()  # 关闭 writer
    print("write tfrecord successful")

def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("The directory was created successfully")
    else:
        print("directory already exists")
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


# 解析 tfrecords 文件    
def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)  # 生成一个先入先出的队列，用来读取数据
    reader = tf.TFRecordReader()    # 新建一个 reader
    _, serialized_example = reader.read(filename_queue) # 把读出的每个样本保存在 serialized_example 中，并进行解序列化
    features = tf.parse_single_example(serialized_example,
                                       features={
                                               'label': tf.FixedLenFeature([10], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string)
                                               })
    img = tf.decode_raw(features['img_raw'], tf.uint8)  # 将 img_raw 字符串转换为 8 位无符号整型
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label

def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size = num,
                                                    num_threads = 2,
                                                    capacity = 1000,
                                                    min_after_dequeue = 700)    # 随机读取一个 batch 的数据
    return img_batch, label_batch

def main():
    generate_tfRecord()
    
if __name__ == '__main__':
    main()