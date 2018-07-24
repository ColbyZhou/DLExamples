# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 17:40:24 2018

@author: zhouqiang02
"""

import numpy as np

def gen_data(raw_inputs, raw_labels, embed_dim, batch_size,
             num_steps_k1, num_steps_k2):
    
    batch_len = len(raw_inputs) // batch_size
    
    batched_inputs = np.zeros([batch_size, batch_len])
    batched_labels = np.zeros([batch_size, batch_len])
    for i in range(batch_size):
        s = i * batch_len
        e = (i + 1) * batch_len
        batched_inputs[i] = raw_inputs[s:e]
        batched_labels[i] = raw_labels[s:e]
        
    # ensure last data has length `num_steps_k2`
    act_batch_len = batch_len - (num_steps_k2 - num_steps_k1)
    size = act_batch_len // num_steps_k1

    for i in range(size):
        
        start_idx = i * num_steps_k1
        end_idx = start_idx + num_steps_k2
                
        # dim [batch_size, num_steps_k2]
        input_list = batched_inputs[:, start_idx : end_idx]
        label_list = batched_labels[:, start_idx : end_idx]

        yield (input_list, label_list)

