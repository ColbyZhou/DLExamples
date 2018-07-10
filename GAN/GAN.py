# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 19:30:31 2018

@author: zhouqiang02
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
from discriminator import Discriminator
from generator import Generator

# real data distribution
# Gaussian
def sample_true_data(num):
    mu = [1, 1]
    sigma = [[1, 0.5], [1.5, 3.5]]
    true_data = np.random.multivariate_normal(mean = mu, cov = sigma, size = num)
    return true_data

# circle
def sample_true_data_1(num):
    # x ~ [-1, 1]
    x = (np.random.rand(num) - 0.5) * 2
    # y ~ +-sqrt(1 - x**2)
    y = np.random.choice([-1, 1], num) * (np.sqrt(1 - x ** 2))
    #data = np.array([(a, b) for a, b in zip(x, y)])
    data = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    return data

# noise z for G
def sample_noise(num, dim):
    z = np.random.normal(0, 1, size = (num, dim))
    return z

def plot_true_data_and_noise():
    # plot true data & noise sample
    true_data = sample_true_data_func(num = 1000)
    noise_data = sample_noise(1000, dim)
    #fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,8))
    plt.figure()
    plt.plot(*(true_data).T, '.', label = "real data")
    plt.legend()
    plt.savefig(os.path.join(img_dir, "real_data.jpg"))

    plt.figure()
    plt.plot(*(noise_data).T, '*', label = "noise data")
    plt.legend()
    plt.savefig(os.path.join(img_dir, "noise_data.jpg"))

def plot_gen_data(final_g_data):
    final_g_data = np.array(final_g_data)
    plt.figure()
    plt.plot(*(final_g_data[0]).T, '.', label = "Generator Data epoc " + str(i))
    plt.legend()
    plt.savefig(os.path.join(img_dir, str(i) + ".jpg"))
    print("save generated imgs, " + str(i))

def plot_loss_fig(D_loss_list, G_loss_list):
    plt.figure()
    plt.plot(range(len(D_loss_list)), D_loss_list, label = 'Discriminator Loss')
    plt.legend()
    plt.savefig(os.path.join(img_dir, "d_loss.jpg"))

    plt.figure()
    plt.plot(range(len(G_loss_list)), G_loss_list, label = "Generator Loss")
    plt.legend()
    plt.savefig(os.path.join(img_dir, "g_loss.jpg"))

dim = 2
#img_dir = "imgs_Gaussian"
img_dir = "imgs_circle"

#sample_true_data_func = sample_true_data
sample_true_data_func = sample_true_data_1
plot_true_data_and_noise(sample_true_data_func)

# hyperparameters
G_learning_rate = 2e-4
D_learning_rate = 2e-4
batch_size = 1000
z_size = dim
input_size = dim
output_size = input_size

# build graph
Generator = Generator(batch_size, [z_size, 128, 128, output_size],
                                #[tf.tanh, tf.tanh, tf.tanh],
                                [tf.nn.leaky_relu, tf.nn.leaky_relu, None], # for Gaussian data
                                #[tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.tanh], # for circle data
                                scope_name = "Generator")
Discriminator = Discriminator(batch_size, [input_size, 128, 128, 1],
                                #[tf.tanh, tf.tanh, tf.sigmoid],
                                #[tf.nn.relu, tf.nn.relu, tf.nn.sigmoid],
                                [tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.sigmoid],
                                scope_name = "Discriminator")

generator_output = Generator.build()
data_preds, gen_preds = Discriminator.build(generator_output)

G_loss = -tf.reduce_mean(tf.log(gen_preds))
D_loss = -tf.reduce_mean((tf.log(data_preds) + tf.log(1 - gen_preds)))

# G trainer
G_trainer = tf.train.AdamOptimizer(G_learning_rate)
G_step = G_trainer.minimize(G_loss, var_list = Generator.get_var_list())

# D trainer
D_trainer = tf.train.AdamOptimizer(D_learning_rate)
D_step = D_trainer.minimize(D_loss, var_list = Discriminator.get_var_list())

G_loss_list = []
D_loss_list = []

with tf.Session() as sess:
    #graph_writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    epoc_num = 20000
    d_steps = 1
    g_steps = 1
    m = batch_size
    # train proces
    for i in range(epoc_num):
        # 1. d steps for optimize D
        for d in range(d_steps):
            # sample m noise samples {z_1, z_2, ..., z_m} from p_z(z)
            z = sample_noise(m, dim)
            # sample m examples {x_1, x_2, ..., x_m} from p_data(x) (true_data)
            true_samples = sample_true_data_func(num = m)

            # update D
            d_loss, _, d_probs, g_probs = sess.run([D_loss, D_step, data_preds, gen_preds],
                    feed_dict = {
                    Generator.get_input_layer_tensor() : z, # p_z
                    Discriminator.get_input_layer_tensor() : true_samples, # true data
                    })
            G_loss_list.append(d_loss)

        # 2. g steps for optimize G
        for g in range(g_steps): 
            # sample m noise samples {z_1, z_2, ..., z_m} from p_z(z)
            z = sample_noise(m, dim)
            # sample m examples {x_1, x_2, ..., x_m} from p_data(x) (true_data)
            true_samples = sample_true_data_func(num = m)
            zero_data_for_G = np.zeros((m, dim))

            # update G
            g_loss, _, g_data = sess.run([G_loss, G_step, generator_output],
                    feed_dict = {
                    Generator.get_input_layer_tensor() : z, # p_z
                    Discriminator.get_input_layer_tensor() : zero_data_for_G, # zero data or true data?
                    })
            D_loss_list.append(g_loss)

        if i % 100 == 0:
            print(str(i) + " of " + str(epoc_num) + ", " + "{:.2f}".format(1.0 * i / epoc_num) + "%")

        if (i + 1) % 1000 == 0 or i == 0:
            z = sample_noise(m, dim)
            final_g_data = sess.run([generator_output],
                    feed_dict = {
                    Generator.get_input_layer_tensor() : z, # p_z
                    })
            plot_gen_data(final_g_data)

plot_loss_fig(D_loss_list, G_loss_list)