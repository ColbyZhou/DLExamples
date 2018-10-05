#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: zhouqiang02
# breif:

import sys
import os
import urlparse
import urllib
import sys
import json
import os
import re
import urllib2

import numpy as np
import tensorflow as tf

lr = 1e-2

# Create Variables In PS

W = tf.get_variable('W', [], dtype = tf.float32, initializer=tf.zeros_initializer)
b = tf.get_variable('b', [], dtype = tf.float32, initializer=tf.zeros_initializer)

X = tf.placeholder(dtype = tf.float32, shape = [], name = "input_x")
Y = tf.placeholder(dtype = tf.float32, shape = [], name = "label_y")

loss = tf.square(tf.subtract(tf.add(tf.multiply(W, X), b), Y))

train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# Simulate Train Data
train_data_X = np.linspace(-1, 1, 100)
train_data_Y = 2 * train_data_X + 10 + np.random.randn(*train_data_X.shape) * 0.1

#with tf.Session() as sess:
with tf.Session('grpc://localhost:2222') as sess:
    with tf.device('/job:worker/task:0'):

        print "here"
        sess.run(tf.global_variables_initializer())

        # Start Training Process
        print "start training..."
        for i in range(20):
            print "epoc: " + str(i)
            for x, y in zip(train_data_X, train_data_Y):

                _, cur_loss = sess.run([train_op, loss], feed_dict = {X: x, Y: y})

                print "loss: " + str(cur_loss)

        _W, _b = sess.run([W, b])

    print "result:"
    print _W
    print _b

