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
from scipy.misc import imsave
import discriminator
import generator

# real data distribution
# circle
def sample_true_data_circle(num):
    # x ~ [-1, 1]
    x = (np.random.rand(num) - 0.5) * 2
    # y ~ +-sqrt(1 - x**2)
    y = np.random.choice([-1, 1], num) * (np.sqrt(1 - x ** 2))
    data = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    return data

# Gaussian
def sample_true_data_Gaussian(num):
    mu = [1, 1]
    sigma = [[1, 0.5], [1.5, 3.5]]
    true_data = np.random.multivariate_normal(mean = mu, cov = sigma, size = num)
    return true_data

# MNIST
mnist = None
def sample_true_data_mnist(num):
    if mnist != None:
        x_value, label = mnist.train.next_batch(num)
        return x_value

# noise z for G
def sample_noise(num, dim):
    z = np.random.normal(0, 1, size = (num, dim))
    return z

def plot_true_data_and_noise(img_dir, sample_true_data_func, dim):
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

def plot_gen_data(img_dir, gen_data, i):
    gen_data = np.array(gen_data)
    plt.figure()
    plt.plot(*(gen_data[0]).T, '.', label = "Generator Data epoc " + str(i))
    plt.legend()
    plt.savefig(os.path.join(img_dir, str(i) + ".jpg"))
    print("save generated imgs, " + str(i))

def save_gen_img_data(img_dir, gen_data, i,
        img_height = 28, img_width = 28, grid = (8, 8), pad = 5):

    gen_data = np.array(gen_data)[0]
    img_data = gen_data.reshape((gen_data.shape[0], img_height, img_width))
    img_h, img_w = img_data.shape[1], img_data.shape[2]

    grid_h = grid[0] * (img_h + pad) - pad
    grid_w = grid[1] * (img_w + pad) - pad

    grid_data = np.zeros((grid_h, grid_w), dtype = np.uint8)
    for idx, data in enumerate(img_data):
        if idx >= grid[0] * grid[1]:
            break

        img = (data * 255).astype(np.uint8)

        row = idx // grid[1]
        col = idx % grid[1]

        row_start = row * (img_h + pad)
        row_end = row_start + img_h
        col_start = col * (img_w + pad)
        col_end = col_start + img_w

        grid_data[row_start:row_end, col_start:col_end] = img
    imsave(os.path.join(img_dir, str(i) + ".jpg"), grid_data)
    print("save generated imgs, " + str(i))

def plot_loss_fig(img_dir, D_loss_list, G_loss_list):
    plt.figure()
    plt.plot(range(len(D_loss_list)), D_loss_list, label = 'Discriminator Loss')
    plt.legend()
    plt.savefig(os.path.join(img_dir, "d_loss.jpg"))

    plt.figure()
    plt.plot(range(len(G_loss_list)), G_loss_list, label = "Generator Loss")
    plt.legend()
    plt.savefig(os.path.join(img_dir, "g_loss.jpg"))

def GAN(data_name = "circle"):

    # hyperparameters
    G_learning_rate = 2e-4
    D_learning_rate = 2e-4
    batch_size = 1000
    epoc_num = 20000
    d_steps = 1
    g_steps = 1

    dim = 0
    ori_z = None
    if data_name == "circle":
        dim = 2
        z_size = dim
        input_size = dim
        output_size = input_size
        img_dir = "imgs_circle"
        sample_true_data_func = sample_true_data_circle
        plot_true_data_and_noise(img_dir, sample_true_data_func, dim)
        g_dim_list = [z_size, 128, 128, output_size]
        g_act_list = [tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.tanh]
    elif data_name == "Gaussian":
        dim = 2
        z_size = dim
        input_size = dim
        output_size = input_size
        img_dir = "imgs_Gaussian"
        sample_true_data_func = sample_true_data_Gaussian
        plot_true_data_and_noise(img_dir, sample_true_data_func, dim)
        g_dim_list = [z_size, 128, 128, output_size]
        g_act_list = [tf.nn.leaky_relu, tf.nn.leaky_relu, None]
    elif data_name == "mnist":

        # increase epoc_num for mnist
        batch_size = 250
        epoc_num = 120000

        img_height = 28
        img_width = 28
        dim = img_height * img_width
        z_size = dim
        input_size = dim
        output_size = input_size
        img_dir = "imgs_mnist"
        global mnist
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        sample_true_data_func = sample_true_data_mnist
        g_dim_list = [z_size, 128, 128, output_size]
        g_act_list = [tf.nn.leaky_relu, tf.nn.leaky_relu, tf.sigmoid]
        ori_z = sample_noise(batch_size, dim)

    d_dim_list = [input_size, 128, 128, 1]
    d_act_list = [tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.sigmoid]

    # build graph
    Generator = generator.Generator(batch_size, g_dim_list, g_act_list, scope_name = "Generator")
    Discriminator = discriminator.Discriminator(batch_size, d_dim_list, d_act_list, scope_name = "Discriminator")

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
                if data_name == "mnist":
                    z = ori_z
                gen_data = sess.run([generator_output],
                        feed_dict = {
                        Generator.get_input_layer_tensor() : z, # p_z
                        })
                if data_name == "mnist":
                    save_gen_img_data(img_dir, gen_data, i + 1, img_height, img_width)
                else:
                    plot_gen_data(img_dir, gen_data, i + 1)

    plot_loss_fig(img_dir, D_loss_list, G_loss_list)

if __name__ == '__main__':
    #GAN("circle")
    #GAN("Gaussian")
    GAN("mnist")