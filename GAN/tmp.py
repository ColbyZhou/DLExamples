# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:45:22 2018

@author: zhouqiang02
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from MLP import MultiLayerPerceptron
from test import Test

sess = tf.Session()

"""
mu = [1, 1]
#sigma = [[1, 0.5], [1.5, 1.5]]
sigma = [[1, 0], [0, 1]]
data = np.random.multivariate_normal(mean = mu,cov = sigma, size = 2000)
x, y = data.T
#plt.plot(x, y, '.')
#plt.axis('scaled')
#plt.show()

z = np.random.uniform(-10, 10, size = (10, 2))
print(z.shape)
print(z)

print(data.shape)

with tf.variable_scope('scope_name'):
    label_placeholder = tf.placeholder(tf.float32,
                    shape = [2, 2],
                    name = "label_placeholder")
    print(label_placeholder.name)
    label_placeholder1 = tf.placeholder(tf.float32,
                    shape = [2, 2],
                    name = "label_placeholder")
    print(label_placeholder1.name)

#"""

"""
dim = 2
batch_size = 100
input_size = dim
output_size = input_size

Generator = MultiLayerPerceptron(batch_size, [input_size, 128, 128, output_size],
                                                 [tf.tanh, tf.tanh, tf.tanh], 0.1)

G_output = Generator.get_output_layer_tensor()
print(G_output.shape)
# feed G_output as extra input tensor for Discriminator
zeros = tf.zeros((G_output.shape[0], 1))
print(zeros.shape)
print(sess.run(zeros))
Discriminator = MultiLayerPerceptron(batch_size, [input_size, 128, 128, 1],
                                                 [tf.tanh, tf.tanh, tf.sigmoid], 0.1,
                                                 extra_tensor_input = G_output)

# Generator's output label are all zero
Discriminator.get_softmax_cross_entropy_loss(extra_data_label = zeros)
"""

"""

Generator.initialize(sess)
Discriminator.initialize(sess)

m = batch_size
z = np.random.uniform(-10, 10, size = (m, dim))
g_output = Generator.forward(input = z)

print(z)
print(g_output)
var_a = Generator.get_var_list_by_collection()
var_b = Generator.get_var_list()
print(var_a)
print(var_b)
"""

"""
tf_rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units = 10)
tf_rnn_cell2 = tf.contrib.rnn.BasicRNNCell(num_units = 10)

print(tf_rnn_cell.name)
print(tf_rnn_cell2.name)

t1 = Test()
t2 = Test()
"""
"""
a = np.array(
    [[1,2,3,4],
     [5,6,7,8],
     [9,3,4,5]])

b = np.array(
    [[3,2,1,0],
     [5,6,7,7],
     [4,5,6,3],
     [2,4,5,1],
     [3,4,1,8]])
a = tf.convert_to_tensor(a, dtype = tf.int32)
b = tf.convert_to_tensor(b, dtype = tf.int32)
print(a)
print(b.shape)

print(b)
print(a.shape)
c = tf.concat([a, b], 0)
print(c)
print(c.shape)

print(sess.run(c))
"""

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
    data = np.array([(a, b) for a, b in zip(x, y)])
    return data

true_data = sample_true_data(1000)
data = sample_true_data_1(1000)
print(true_data.shape)
print(data)
print(data.shape)