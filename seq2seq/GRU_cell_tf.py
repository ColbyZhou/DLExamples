# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 21:04:40 2018

@author: zhouqiang02
"""

import tensorflow as tf

class MyGRUCell:
    """
        GRU by my own
    """
    def __init__(self, state_size, input_size):
        """
        state_size: hidden layer size
        input_size: input_size
        """
        self.state_size = state_size
        self.input_size = input_size
        
        self.W_r = tf.get_variable(
                "W_r", shape = [self.state_size + self.input_size, self.state_size],
                #initializer=tf.constant_initializer(0.1)
                )
        self.b_r = tf.get_variable(
                "b_r", shape = [self.state_size],
                initializer=tf.constant_initializer(0.1)
                )
        self.W_z = tf.get_variable(
                "W_z", shape = [self.state_size + self.input_size, self.state_size],
                #initializer=tf.constant_initializer(0.1)
                )
        self.b_z = tf.get_variable(
                "b_z", shape = [self.state_size],
                initializer=tf.constant_initializer(0.1)
                )
        self.W_h = tf.get_variable(
                "W_h", shape = [self.state_size + self.input_size, self.state_size],
                #initializer=tf.constant_initializer(0.1)
                )
        self.b_h = tf.get_variable(
                "b_h", shape = [self.state_size],
                initializer=tf.constant_initializer(0.1)
                )
    def _gate(self, W, b, h, input):
        """
        construct a gate
        """
        gate = tf.sigmoid(
                tf.add(
                    tf.matmul(
                            tf.concat([h, input], 1),
                            W),
                    b)
                )
        return gate

    def one_time_step(self, input, h):
        """
            input: [batch_size, input_size] tensor
            h: [batch_size, state_size] tensor
            C: [batch_size, state_size] tensor
            
            C: [batch_size, state_size] tensor
            h: [batch_size, state_size] tensor
            
            given inputs and last C & h, return new C & h
        """
        # dim: [batch_size, state_size]
        # reset gate
        r = self._gate(self.W_f, self.b_f, h, input)
        # update gate
        z = self._gate(self.W_i, self.b_i, h, input)

        # [batch_size, state_size]
        h_hat = tf.tanh(
                tf.add(
                    tf.matmul(
                            tf.concat([
                                    tf.multiply(r, h), 
                                    input], 1),
                            self.W_h),
                    self.b_h)
                )
        h_new = tf.add(
                tf.multiply(
                        tf.subtract(1, z),
                        h),
                tf.multiply(z, h_hat)
                )
        return h_new
    
    
