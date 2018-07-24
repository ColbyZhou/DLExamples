# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:57:12 2018

@author: zhouqiang02
"""

import tensorflow as tf

class MyRNNCell:
    """
        RNN by my own
    """
    def __init__(self, state_size, input_size):
        """
        state_size: hidden layer size
        input_size: input_size
        """
        self.state_size = state_size
        self.input_size = input_size
        self.W = tf.get_variable(
                name = 'W',
                shape = [self.state_size + self.input_size, self.state_size],
                #initializer=tf.constant_initializer(0.1)
                #intializer = tf.random_normal_initializer(mean = 0.1, stddev = 1.0)
                )
        self.b = tf.get_variable(
                name = 'b',
                shape = [self.state_size],
                initializer=tf.constant_initializer(0.1))
            
    def one_time_step(self, input, state):
        """
            input: [batch_size, input_size] tensor
            state: [batch_size, state_size] tensor
            output: [batch_size, state_size] tensor
            
            given inputs and last state, return output
            h(t) = tanh(W*h(t-1) + U*X + b)
        """
        
        output = tf.tanh(
                    tf.add(
                        tf.matmul(
                                tf.concat([input, state], 1),
                                self.W),
                        self.b))
        return output