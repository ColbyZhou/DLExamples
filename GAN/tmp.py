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
import re
import weakref
import collections

"""
mu = [1, 1]
#sigma = [[1, 0.5], [1.5, 1.5]]
sigma = [[1, 0], [0, 1]]
data = np.random.multivariate_normal(mean = mu,cov = sigma, size = 2000)
x, y = data.T
plt.plot(x, y, '.')
plt.axis('scaled')
plt.show()

z = np.random.uniform(-10, 10, size = (10, 2))
print(z.shape)
print(z)
"""

#"""
dim = 2
batch_size = 100
input_size = dim
output_size = input_size

Generator = MultiLayerPerceptron(batch_size, [input_size, 128, 128, output_size],
                                                 [tf.tanh, tf.tanh, tf.tanh], 0.1)

Discriminator = MultiLayerPerceptron(batch_size, [input_size, 128, 128, 1], [tf.tanh, tf.tanh, tf.sigmoid], 0.1)

"""
Generator.initialize()
#Discriminator.initialize()

m = batch_size
z = np.random.uniform(-10, 10, size = (m, dim))
g_output = Generator.forward(input = z)

#print(z)
#print(g_output)
#"""

tf_rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units = 10)
tf_rnn_cell2 = tf.contrib.rnn.BasicRNNCell(num_units = 10)

print(tf_rnn_cell.name)
print(tf_rnn_cell2.name)

t1 = Test()
t2 = Test()


