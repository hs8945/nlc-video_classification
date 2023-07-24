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



def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
  
def ENCL(input_tensor,y):
    # #Non-local softmax归一化,embedded gaussian模式
    # theta = conv2d(input_tensor, channels, channels // 2, 1)
    theta_w_conv = weight_variable([1,1,3,3])
    theta_b_conv = bias_variable([3])
    theta = conv2d(cnn_data,theta_w_conv)+theta_b_conv
    theta = tf.reshape(theta, shape=[-1, 256, 3])



    phi_w_conv = weight_variable([1,1,3,3])
    phi_b_conv = bias_variable([3])
    phi = conv2d(cnn_data,phi_w_conv)+phi_b_conv
    phi = tf.reshape(phi, shape=[-1, 256, 3])


    f = tf.matmul(theta, phi, transpose_b=True)  #256*256
    phi_shape = phi.get_shape().as_list()
    f = tf.reshape(f, shape=[-1, 256, phi_shape[1]])    
    f = tf.nn.softmax(f, axis=-1)



    g_w_conv = weight_variable([1,1,3,3])
    g_b_conv = bias_variable([3])
    g = conv2d(cnn_data,g_w_conv)+g_b_conv
    g = tf.reshape(g, shape=[-1, 256,3])

    y_new = tf.matmul(g,f,transpose_a=True,transpose_b=True)
    g = tf.reshape(g,shape=[-1,256,1,3])


# #ENL
    g_new_w_conv = weight_variable([1,1,3,1])
    g_new_b_conv = bias_variable([1])
    g_new = conv2d(g,g_new_w_conv)+g_new_b_conv
    g_new = tf.reshape(g_new,shape=[-1,1,256])
    g_new = tf.nn.softmax(g_new,axis=-1)
    g = tf.reshape(g,shape=[-1,256,3])
    g_new = tf.matmul(g_new,g)

    g_new = tf.reshape(g_new,shape=[-1,1,1,3])# 1,1,C
    y_m = tf.reshape(y_new,shape=[-1,256,1,3])
    output_tensor = g_new + y_m
    return output_tensor
    
def ecanet_layer(x): 
        k_size = 3
        squeeze = tf.expand_dims(x,axis=1)
        squeeze = tf.transpose(squeeze,[0,2,1])

        print('x dim',x.shape)
        attn = tf.layers.Conv1D(filters=1,
            kernel_size=k_size,
            padding='same',
            use_bias=False)(squeeze)    #1024x1   squeeze
        print('attn dim',attn.shape)
        attn = tf.expand_dims(tf.transpose(attn, [0, 2, 1]), 3)
        attn = tf.squeeze(attn)
        x = tf.squeeze(x)
        print('atten dim',attn.shape)
        attn = tf.math.sigmoid(attn)
        scale = x*attn  
        print('scale dim',scale.shape)
        return scale
      
def rnn_model_2(x):
    """RNN (LSTM or GRU) model for image"""
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input_audio])
    x = tf.split(x, n_steps_audio, 0)     # 按行分，n_steps_visual
    

    with tf.variable_scope('audio'):
        lstm_cell = rnn.BasicLSTMCell(n_hidden_audio, forget_bias=1.0)
        audio_outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        
    return audio_outputs
  
  
def context_gating(name, input_layer, add_batch_norm=True):
    with tf.variable_scope(name + 'context_gating'):

        input_dim = input_layer.get_shape().as_list()[1]

        gating_weights = tf.get_variable("gating_weights", [input_dim, input_dim],
                                         initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))

        gates = tf.matmul(input_layer, gating_weights)

        if add_batch_norm:
            gates = slim.batch_norm(gates, center=True, scale=True, is_training=True,
                                    scope="gating_bn")
        else:
            gating_biases = tf.get_variable("gating_biases", [input_dim],
                                            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
            gates += gating_biases

        gates = tf.nn.sigmoid(gates)

        activation = tf.multiply(input_layer, gates)

    return activation

    
  # NetVLAD 2016 CVPR NetVLAD: CNN architecture for weakly supervised place recognition
def NetVLAD(name,reshaped_input, feature_size=512, max_samples=6, cluster_size=32,output_dim=256,  group=128, add_batch_norm=True):

    with tf.variable_scope(name + 'forward'):
        #####################################################################################################################
 
        reshaped_input = tf.reshape(reshaped_input, [-1, feature_size])     
        
        cluster_weights = tf.get_variable("cluster_weights",[feature_size, cluster_size],
                                            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size))
                                          )
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if add_batch_norm:
            activation = slim.batch_norm( activation,center=True,scale=True,is_training=True, scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",[cluster_size],initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
            activation += cluster_biases
            activation = tf.nn.softmax(activation)                            
        activation = tf.reshape(activation,[-1, max_samples, cluster_size])  
        #################################################################################################################

        a_sum = tf.reduce_sum(activation,1,keep_dims=True) 
        cluster_weights2 = tf.get_variable("cluster_weights2", [1, feature_size,  cluster_size],
                                           initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size))
                                           )
        
        
        a = tf.multiply(a_sum,cluster_weights2) 
        
        #######################################################################################################################
  

        activation = tf.transpose(activation,perm=[0,2,1])                                    # (?, 64, 6)
        reshaped_input = tf.reshape(reshaped_input,[-1,max_samples, feature_size])  # (?, 6, 512)

        vlad = tf.matmul(activation,reshaped_input)                                           # (?, 64, 512)

        vlad = tf.transpose(vlad,perm=[0,2,1])                                                # (?, 512, 64)

 
        vlad = tf.subtract(vlad,a)
        
        #######################################################################################################################
        vlad = tf.nn.l2_normalize(vlad,1)
        vlad = tf.reshape(vlad,[-1, cluster_size*feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)
        vlad = slim.batch_norm( vlad,center=True,scale=True,is_training=True, scope="vlad_bn")

        
    
        hidden1_weights = tf.get_variable("hidden1_weights",[cluster_size*feature_size, output_dim],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size))
                                          )
        vlad = tf.matmul(vlad, hidden1_weights)

    return vlad    

# NetVLAD的改进版
# 来自论文：NeXtVLAD: An Efficient Neural Network to Aggregate Frame-level Features for Large-scale Video Classification ECCV2018 2nd Workshop
# 增加了一个attention
def NeXtVLAD(name, input_tensor, feature_size=100, max_samples=30, cluster_size=32, output_dim=256, group=128,
             add_batch_norm=True):
    with tf.variable_scope(name + 'forward'):
        # 第1步：升维 N -> 2N
        expansion = 2
        #         input_tensor = slim.fully_connected(input_tensor, expansion * feature_size, activation_fn=None, weights_initializer=slim.variance_scaling_initializer())
        input_tensor = slim.fully_connected(input_tensor, expansion * feature_size, activation_fn=tf.nn.leaky_relu,
                                            weights_initializer=slim.variance_scaling_initializer())

        ################################################################################################
        # 第2步： attention 注意力函数
        groups = group

        attention = slim.fully_connected(input_tensor, groups, activation_fn=tf.nn.sigmoid,
                                         weights_initializer=slim.variance_scaling_initializer())

        attention = tf.reshape(attention, [-1, max_samples * groups, 1])
        print('NeXtVLAD attention shape', attention.shape)
        ################################################################################################
        # softmax 软分配
        new_feature_size = expansion * feature_size // groups
        #         new_feature_size = expansion * feature_size/groups

        cluster_weights = tf.get_variable("cluster_weights",
                                          [expansion * feature_size, groups * cluster_size],
                                          initializer=slim.variance_scaling_initializer()
                                          )

        reshaped_input = tf.reshape(input_tensor, [-1, expansion * feature_size])
        activation = tf.matmul(reshaped_input, cluster_weights)

        activation = slim.batch_norm(activation, center=True, scale=True, is_training=True, scope="cluster_bn1")

        activation = tf.reshape(activation, [-1, max_samples * groups, cluster_size])
        activation = tf.nn.softmax(activation, dim=-1)
        ###################################################################################################
        # 公式（5）的前半部分 ag * agk
        activation = tf.multiply(activation, attention)

        ###################################################################################################
        # 减法

        a_sum = tf.reduce_sum(activation, 1, keep_dims=True)

        # 聚类中心，也是学出来的
        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, new_feature_size, cluster_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        a = tf.multiply(a_sum, cluster_weights2)
        ###################################################################################################
        # 被减法
        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(input_tensor, [-1, max_samples * groups, new_feature_size])

        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        ##################################################################################################
        # 被减法 - 减法        公式（6）完成
        vlad = tf.subtract(vlad, a)

        ###################################################################################################
        # 归一化
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, cluster_size * new_feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)  ##保留尾部归一化
        vlad = slim.batch_norm(vlad, center=True, scale=True, is_training=True, scope="vlad_bn")

        # 主要为了是输出相应维度的向量
        hidden1_weights = tf.get_variable("hidden1_weights1", [cluster_size * new_feature_size, output_dim],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size))
                                          )

        vlad = tf.matmul(vlad, hidden1_weights)  # (?, 256)
        return vlad

    # NeXtVLAD的改进版


# herad +ReLU,  tail+L2
def NNeXtVLAD(name, inputs, feature_size=512, max_samples=6, cluster_size=32, output_dim=256, group=128,
              add_batch_norm=True):
    with tf.variable_scope(name + 'forward'):
        expansion = 2
        print('inputs before dim', inputs.shape)  
        # 按行shuffle
        #         inputs = tf.transpose(inputs,[1,0,2])   
        #         inputs = tf.random_shuffle(inputs)
        #         inputs = tf.transpose(inputs,[1,0,2])
        #         inputs = tf.random_shuffle(inputs)  
        inputs = slim.fully_connected(inputs, expansion * feature_size, activation_fn=tf.nn.relu,
                                      weights_initializer=slim.variance_scaling_initializer())
        #         inputs     = slim.fully_connected(inputs, expansion * feature_size, activation_fn=tf.nn.leaky_relu, weights_initializer=slim.variance_scaling_initializer())
        print('inputs after dim', inputs.shape)  # 11*8192
        ################################################################################################
        # 2： attention
        groups = group  # 16
        result = slim.fully_connected(inputs, groups, activation_fn=tf.nn.sigmoid,
                                      weights_initializer=slim.variance_scaling_initializer())
        print('results dims', result.shape)  # 11*16
        attention = tf.reshape(result, [-1, max_samples * groups, 1])  
        print('attention dims', attention.shape)
        ################################################################################################
        reshaped_input = tf.reshape(inputs, [-1, expansion * feature_size])  # 11*8192
        new_feature_size = expansion * feature_size // groups  
        cluster_weights = tf.get_variable("cluster_weights",
                                          [expansion * feature_size, groups * cluster_size],
                                          initializer=slim.variance_scaling_initializer()
                                          )
        print('cluster_weights shape', cluster_weights.shape)  # 8192*512
        activation = tf.matmul(reshaped_input, cluster_weights)
        print('activation shape before BN', activation.shape)  # 11*512
        activation = slim.batch_norm(activation, center=True, scale=True,
                                     is_training=True, scope="cluster_bn")

        activation = tf.reshape(activation, [-1, max_samples * groups, cluster_size])
        print('activation shape after BN', activation.shape)  # 176*32
        activation = tf.nn.softmax(activation, dim=-1)
        ###################################################################################################
        activation = tf.multiply(activation, attention)  # cluster center
        print('activation shape multiply attention', activation.shape)

        ###################################################################################################
        a_sum = tf.reduce_sum(activation, 1, keep_dims=True)
        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, new_feature_size, cluster_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        a = tf.multiply(a_sum, cluster_weights2)
        ###################################################################################################
        activation = tf.transpose(activation, perm=[0, 2, 1])  # 32*176

        reshaped_input = tf.reshape(inputs, [-1, max_samples * groups, new_feature_size])

        vlad = tf.matmul(activation, reshaped_input)  # 32*512
        vlad = tf.transpose(vlad, perm=[0, 2, 1])  # 512*32
        ###################################################################################################
        vlad = tf.subtract(vlad, a)

        ###################################################################################################
        vlad = tf.nn.l2_normalize(vlad, 1)  # intra-normalization
        vlad = tf.reshape(vlad, [-1, cluster_size * new_feature_size])  # 16384
        vlad = tf.nn.l2_normalize(vlad, 1)  # L2-normalization

        vlad = slim.batch_norm(vlad, center=True, scale=True, is_training=True, scope="vlad_bn")
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [cluster_size * new_feature_size, output_dim],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size))
                                          )

        vlad = tf.matmul(vlad, hidden1_weights)

        return vlad
