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
  
  
def new_context_gating(name, input_layer):
     with tf.variable_scope(name+'new_context_gating'):
            
#         vlad = input_layer.slim.dropout(vlad, keep_prob=0.5, is_training=self.is_training, scope="vlad_dropout")

        vlad_dim = input_layer.get_shape().as_list()[1]
#         vlad = slim.dropout(input_layer, keep_prob=dropout_keep_prob, is_training=True, scope="vlad_dropout1")
        print("VLAD dimension", vlad_dim)
        hidden1_weights_1 = tf.get_variable("hidden1_weights_1",
                                          [vlad_dim, 2048],    #1024x2048
                                          initializer=slim.variance_scaling_initializer())
#                                             initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(vlad_dim)))

        activation = tf.matmul(input_layer, hidden1_weights_1) #2048
#         print('activation dimention',activation.get_shape().as_list()[1])
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=True,
            scope="hidden1_bn",
            fused=False)
        gating_weights_1 = tf.get_variable("gating_weights_1",
#                                            [hidden1_size, hidden1_size // 8],
#                                            [2048,2048],
                                           [2048,256],
                                           initializer=slim.variance_scaling_initializer())
#                                            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(vlad_dim)))

        gates = tf.matmul(activation, gating_weights_1)     
#         print('gates dimention',gates.get_shape().as_list()[1])
#         gates = tf.matmul(input_layer, gating_weights_1)


        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=True,
            activation_fn=slim.nn.relu,
            scope="gating_bn")

        gating_weights_2 = tf.get_variable("gating_weights_2",
                                                                                     [256,2048],
                                           initializer=slim.variance_scaling_initializer()
#                                            initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(vlad_dim))
                                           )
        gates = tf.matmul(gates, gating_weights_2) #2048
#         print('gates dimention',gates.get_shape().as_list()[1])
        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=True,
            activation_fn=slim.nn.relu,
            scope="gating_bn2")

        gates = tf.sigmoid(gates)
#         tf.summary.histogram("final_gates", gates)

        activation = tf.multiply(activation, gates)
#         activation = tf.multiply(input_layer,gates)
        l2_penalty = 1e-8
        
        return activation
