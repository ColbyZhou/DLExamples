# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:48:39 2018

@author: zhouqiang02
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from MLP import MultiLayerPerceptron

# 真实数据分布 Gaussian
mu = 0
sigma = 0.1

p_data = np.random.normal()

dim = 2

batch_size = 100
input_size = dim
output_size = input_size
Generator = MultiLayerPerceptron(batch_size, [input_size, 128, 128, output_size],
                                                 [tf.tanh, tf.tanh, tf.tanh], 0.1)
Discriminator = MultiLayerPerceptron(batch_size, [input_size, 128, 128, 1],
                                                 [tf.tanh, tf.tanh, tf.sigmoid], 0.1)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    
    epoc_num = 1
    k_steps = 5
    m = batch_size
    # train proces
    for i in range(epoc_num):
        # k steps for optimize D
        for k in range(k_steps):
            
            # sample m noise samples {z_1, z_2, ..., z_m} from p_z(z)
            z = np.random.uniform(-10, 10, size = (m, dim))
            
            # sample m examples {x_1, x_2, ..., x_m} from p_data(x)
            
            g_output = Generator.forward(input = z)
            
            # update D
            #sess.run([train_op], feed_dict = {})
        
        # sample m noise samples {z_1, z_2, ..., z_m} from p_z(z)
        
        # update G
        
        
        