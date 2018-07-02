# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:48:39 2018

@author: zhouqiang02
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyloy as plt

# 真实数据分布 Gaussian
mu = 0
sigma = 0.1

p_data = np.random.normal()

batch_size = 10
dim = 2

# define G

W = tf.get_variable('W', shape = [dim], initializer = tf.constant_initializer(0.01))
b = tf.get_variable('b', shape = [], initializer, tf.constant_initializer(0.01))

X = tf.placeholder(dtype = tf.float32, shape = [batch_size, dim], name = "X")

y = W * X + b
loss = tf.reduce_mean(tf.square(y - x))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)



with tf.Session() as sess:
    init = tf.global_variables_initializer()
    
    epoc_num = 1
    k_steps = 5
    m = 10
    # train proces
    for i in range(epoc_num):
        # k steps for optimize D
        for k in range(k_steps):
            
            # sample m noise samples {z_1, z_2, ..., z_m} from p_z(z)
            z = np.random.uniform(-10, 10)
            
            # sample m examples {x_1, x_2, ..., x_m} from p_data(x)
            
            # update D
            #sess.run([train_op], feed_dict = {})
        
        # sample m noise samples {z_1, z_2, ..., z_m} from p_z(z)
        
        # update G
        
        
        