# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:29:42 2018

@author: zhouqiang02
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
from scipy.misc import imsave
import con_discriminator
import con_generator

# real data distribution
# MNIST
mnist = None
def sample_true_data_mnist(num):
    if mnist != None:
        x_value, x_label = mnist.train.next_batch(num)
        x_label = x_label.astype(np.int32)
        return x_value, x_label

# noise z for G
def sample_noise(num, dim):
    z = np.random.normal(0, 1, size = (num, dim))
    return z

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
    d_steps = 1
    g_steps = 1

    dim = 0

    # increase epoc_num for mnist
    batch_size = 250
    epoc_num = 120000

    img_height = 28
    img_width = 28
    dim = img_height * img_width
    z_size = dim
    input_size = dim
    output_size = input_size
    label_num = 10
    label_embed_dim = 128
    
    img_dir = "imgs_mnist_conGAN"
    global mnist
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    sample_true_data_func = sample_true_data_mnist
    g_dim_list = [z_size, 128, 128, output_size]
    g_act_list = [tf.nn.leaky_relu, tf.nn.leaky_relu, tf.sigmoid]
    ori_z = sample_noise(batch_size, dim)
    
    ori_labels = np.random.choice(label_num, [batch_size])
    print(ori_labels)

    d_dim_list = [input_size, 128, 128, 1]
    d_act_list = [tf.nn.leaky_relu, tf.nn.leaky_relu, tf.nn.sigmoid]

    # build graph
    Generator = con_generator.ConGenerator(batch_size, g_dim_list, g_act_list,
                        label_num, label_embed_dim, scope_name = "ConGenerator")
    Discriminator = con_discriminator.ConDiscriminator(batch_size, d_dim_list, d_act_list,
                        label_num, label_embed_dim, scope_name = "ConDiscriminator")

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
                true_samples, labels = sample_true_data_func(num = m)

                # update D
                d_loss, _, d_probs, g_probs = sess.run([D_loss, D_step, data_preds, gen_preds],
                        feed_dict = {
                        Generator.get_input_layer_tensor() : z, # p_z
                        Generator.get_input_label_tensor(): labels, # labels for generating fake data
                        Discriminator.get_input_layer_tensor() : true_samples, # true data
                        Discriminator.get_input_label_tensor(): labels, # labels of true data
                        Discriminator.get_generator_output_label_tensor(): labels, # labels for Generator to trick
                        })
                G_loss_list.append(d_loss)

            # 2. g steps for optimize G
            for g in range(g_steps): 
                # sample m noise samples {z_1, z_2, ..., z_m} from p_z(z)
                z = sample_noise(m, dim)
                # sample m examples {x_1, x_2, ..., x_m} from p_data(x) (true_data)
                true_samples, labels = sample_true_data_func(num = m)
                zero_data_for_G = np.zeros((m, dim))

                # update G
                g_loss, _, g_data = sess.run([G_loss, G_step, generator_output],
                        feed_dict = {
                        Generator.get_input_layer_tensor() : z, # p_z
                        Generator.get_input_label_tensor(): labels, # labels for generating fake data
                        Discriminator.get_input_layer_tensor() : zero_data_for_G, # zero data or true data?
                        Discriminator.get_input_label_tensor(): labels, # labels of true data
                        Discriminator.get_generator_output_label_tensor(): labels, # labels for Generator to trick
                        })
                D_loss_list.append(g_loss)

            if i % 100 == 0:
                print(str(i) + " of " + str(epoc_num) + ", " + "{:.2f}".format(100.0 * i / epoc_num) + "%")

            if (i + 1) % 1000 == 0 or i == 0:
                z = sample_noise(m, dim)
                z = ori_z
                
                gen_data = sess.run([generator_output],
                        feed_dict = {
                        Generator.get_input_layer_tensor() : z, # p_z
                        Generator.get_input_label_tensor(): ori_labels,
                        })
                save_gen_img_data(img_dir, gen_data, i + 1, img_height, img_width)

    plot_loss_fig(img_dir, D_loss_list, G_loss_list)

if __name__ == '__main__':
    #GAN("circle")
    #GAN("Gaussian")
    GAN("mnist")