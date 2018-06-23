# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 17:40:24 2018

@author: zhouqiang02
"""

import tensorflow as tf
import numpy as np

def gen_data(raw_inputs, raw_labels, embed_dim, batch_size,
             num_steps_k1, num_steps_k2):
    
    batch_len = len(raw_inputs) / batch_size
    
    batched_inputs = np.zeros([batch_size, batch_len], dtype = tf.int32)
    batched_labels = np.zeros([batch_size, batch_len], dtype = tf.int32)
    for i in range(batch_size):
        s = i * batch_len
        e = (i + 1) * batch_len
        batched_inputs[i] = raw_inputs[s:e]
        batched_labels[i] = raw_labels[s:e]
        
    size = batch_len / num_steps_k1

    for i in range(size):
        end_idx = (i + 1) * num_steps_k1
        start_idx = max(0, end_idx - num_steps_k2)
        
        # dim [batch_size, num_steps_k2]
        input_list = batched_inputs[:, start_idx : end_idx]
        label_list = batched_labels[:, start_idx : end_idx]
        
        # dim [batch_size, num_steps_k2, embed_dim]
        input_list = tf.one_hot(input_list, embed_dim)
        # array of num_steps_k2 elements, with dim [batch_size, embed_dim]
        input_list = tf.unstack(input_list, axis = 1)
        
        label_list = tf.one_hot(label_list, embed_dim)
        label_list = tf.unstack(label_list, axis = 1)
        
        yield (input_list, label_list)

