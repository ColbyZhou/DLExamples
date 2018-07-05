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
dim = 2

def sample_true_data(num):
    mu = [1, 1]
    sigma = [[1, 0.5], [1.5, 3.5]]
    true_data = np.random.multivariate_normal(mean = mu, cov = sigma, size = num)
    return true_data

def sample_noise(num, dim):
    z = np.random.uniform(-10, 10, size = (num, dim))
    return z

# plot true data & noise sample
#"""
x, y = sample_true_data(num = 2000).T
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,3))
axes[0].plot(x, y, '.')
axes[1].plot(*(sample_noise(100, dim)).T, '*')
plt.show()
#"""

# hyperparameters
G_learning_rate = 0.1
D_learning_rate = 0.1
batch_size = 1000
input_size = dim
output_size = input_size

# build graph
Generator = MultiLayerPerceptron(batch_size, [input_size, 128, 128, output_size],
                                                 [tf.tanh, tf.tanh, tf.tanh], G_learning_rate,
                                                 scope_name = "Generator")
G_output = Generator.get_output_layer_tensor()
# feed G_output as extra input tensor for Discriminator
Discriminator = MultiLayerPerceptron(batch_size, [input_size, 128, 128, 1],
                                                 [tf.tanh, tf.tanh, tf.sigmoid], D_learning_rate,
                                                 scope_name = "Discriminator",
                                                 extra_tensor_input = G_output)
Discriminator.creat_label_placeholder()

# G loss
ones_label = tf.ones((G_output.shape[0], 1))
# Generator's output label are all ones, in order to fake Discriminator
G_loss = Discriminator.get_softmax_cross_entropy_loss(extra_data_label = ones_label)

# D loss
zeros_label = tf.zeros((G_output.shape[0], 1))
# Generator's output label are all zero, in order to recognize Generator's fake data
D_loss = Discriminator.get_softmax_cross_entropy_loss(extra_data_label = zeros_label)

# G trainer
G_trainer = tf.train.AdamOptimizer(G_learning_rate)
G_step = G_trainer.minimize(G_loss, var_list = Generator.get_var_list())
# D trainer
D_trainer = tf.train.AdamOptimizer(D_learning_rate)
D_step = D_trainer.minimize(D_loss, var_list = Discriminator.get_var_list())

G_loss_list = []
D_loss_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  
    epoc_num = 3
    d_steps = 2
    g_steps = 1
    m = batch_size
    # train proces
    for i in range(epoc_num):
        # 1. d steps for optimize D
        for d in range(d_steps):
            # sample m noise samples {z_1, z_2, ..., z_m} from p_z(z)
            z = sample_noise(m, dim)
            # sample m examples {x_1, x_2, ..., x_m} from p_data(x) (true_data)
            true_samples = sample_true_data(num = m)
            all_ones = np.ones((m,1))
            # update D
            d_loss, _ = sess.run([D_loss, D_step],
                    feed_dict = {
                    Generator.get_input_layer_tensor() : z, # p_z
                    Discriminator.get_input_layer_tensor() : true_samples, # true data
                    Discriminator.get_label_placeholder() : all_ones, # all ones
                    })
            G_loss_list.append(d_loss)

        # 2. g steps for optimize G
        for g in range(g_steps): 
            # sample m noise samples {z_1, z_2, ..., z_m} from p_z(z)
            z = sample_noise(m, dim)
            zero_data_for_G = np.zeros((m, dim))
            #true_samples = sample_true_data(num = m)
            all_zeros = np.zeros((m, 1))
            g_loss, _, g_data = sess.run([G_loss, G_step, G_output],
                    feed_dict = {
                    Generator.get_input_layer_tensor() : z, # p_z
                    Discriminator.get_input_layer_tensor() : zero_data_for_G, # zero data or true data?
                    Discriminator.get_label_placeholder() : all_zeros, # all zeros
                    })
            D_loss_list.append(g_loss)

            if i * g_steps + g % 5 == 0:
                plt.plot(*g_data.T, '.', label = "Generator Data")
                plt.show()


plt.plot(range(len(D_loss_list)), D_loss_list, label = 'Discriminator Loss')
plt.legend()
plt.plot(range(len(G_loss_list)), G_loss_list, label = "Generator Loss")
plt.legend()
plt.show()
