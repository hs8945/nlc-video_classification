import tensorflow as tf
import numpy as np
import h5py
import numpy
from numpy import *
import time
import cv2
import os
import signal
import tensorflow.contrib as tc

import tensorflow.contrib.slim as slim
from module import *


batch_size = 100
tf.reset_default_graph()

n_input_visual = 4096 
n_steps_visual= 11  

n_input_audio=512
n_steps_audio=6

n_input_text = 100
n_steps_text = 30

# 学习率参数
learning_rate =0.01
lr=tf.Variable(learning_rate,trainable=False)
current_lr=learning_rate
learning_rate_decay=0.3

# 迭代参数
epoch = 20
training_iters = 18000  
display_step = 10


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def next_batch(step, batch_size,images,labels):
    """Return the next `batch_size` examples from this data set."""

    index_in_epoch = batch_size
    end = step*index_in_epoch
    start=(step-1)*index_in_epoch
    return images[start:end],  labels[start:end]
  
  
def  evaluate(pred,test_label):
    task_dim=len(numpy.unique(test_label))
    test_label=numpy.array(test_label)
    pred=numpy.array(pred)
    TP=zeros(task_dim)
    TN=zeros(task_dim)
    FN=zeros(task_dim)
    FP=zeros(task_dim)
    Recall=zeros(task_dim)
    Precision=zeros(task_dim)
    F=zeros(task_dim)
    eps=1e-8;
    for i in range(task_dim):
        set_p=numpy.where(test_label==i) # tuple (row_index,column_index), pred is a column vector only using 0 index
        set_p_=numpy.where(pred==i)
        set_n=numpy.where(test_label!=i) # tuple
        set_n_=numpy.where(pred!=i)
        p=len(set_p[0])
        p_=len(set_p_[0])
        n=len(set_n[0])
        n_=len(set_n_[0])
        if p_!=0:
            for j in range(p_):

                if set_p_[0][j] in set_p[0]:
                    TP[i]=TP[i]+1
                if  set_p_[0][j] in set_n[0]:
                    FP[i]=FP[i]+1
        if n_!=0:
            for j in range(n_):
              if set_n_[0][j] in set_n[0]:
                    TN[i]=TN[i]+1
                if  set_n_[0][j] in set_p[0]:
                    FN[i]=FN[i]+1          
        Recall[i]=TP[i]/(TP[i]+FN[i]+eps)
        Precision[i]=TP[i]/(TP[i]+FP[i]+eps)
        F[i]=(2*Recall[i]* Precision[i])/( Precision[i]+Recall[i]+eps);
    micro_p=sum(TP)/(sum(TP)+sum(FP)+eps)
    micro_R=sum(TP)/(sum(TP)+sum(FN)+eps)
    micro_F=(2*micro_p*micro_R)/(micro_R+micro_p+eps)
    macro_f=sum(F)/task_dim
    return micro_F, macro_f

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1   VALID
    # strides
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# #adam
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
# # sgd&momentum
# optimizer = tf.train.MomentumOptimizer(lr,0.9).minimize(cost)
  
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   #初始化所有参数

    for idx in range(epoch):
        step = 1
        print('current lr:',current_lr)
        # 每一轮次都打乱
        num_examples=train_audio.shape[0]
        perm = numpy.arange(num_examples)
        numpy.random.shuffle(perm)
#         if idx > 1:
#             numpy.random.shuffle(perm)
#             print('shuffle')
        
  
        visuals = train_visual[perm]
        audios = train_audio[perm]
        texts = train_text[perm]
        
        labels = train_label[perm]
        
        
        current_lr=current_lr * learning_rate_decay

        lr.assign(current_lr).eval()    # 可变学习率赋值
