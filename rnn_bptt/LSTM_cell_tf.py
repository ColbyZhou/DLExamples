# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:31:10 2018

@author: zhouqiang02
"""

import tensorflow as tf

class MyLSTMCell:
    """
        LSTM by my own
    """
    def __init__(self, state_size, input_size):
        """
        state_size: hidden layer size
        input_size: input_size
        """
        self.state_size = state_size
        self.input_size = input_size
        
        self.W_f = tf.get_variable(
                "W_f", shape = [self.state_size + self.input_size, self.state_size],
                #initializer=tf.constant_initializer(0.1)
                )
        self.b_f = tf.get_variable(
                "b_f", shape = [self.state_size],
                initializer=tf.constant_initializer(0.1)
                )
        self.W_i = tf.get_variable(
                "W_i", shape = [self.state_size + self.input_size, self.state_size],
                #initializer=tf.constant_initializer(0.1)
                )
        self.b_i = tf.get_variable(
                "b_i", shape = [self.state_size],
                initializer=tf.constant_initializer(0.1)
                )
        self.W_o = tf.get_variable(
                "W_o", shape = [self.state_size + self.input_size, self.state_size],
                #initializer=tf.constant_initializer(0.1)
                )
        self.b_o = tf.get_variable(
                "b_o", shape = [self.state_size],
                initializer=tf.constant_initializer(0.1)
                )
        self.W_C = tf.get_variable(
                "W_C", shape = [self.state_size + self.input_size, self.state_size],
                #initializer=tf.constant_initializer(0.1)
                )
        self.b_C = tf.get_variable(
                "b_C", shape = [self.state_size],
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

    def one_time_step(self, input, C, h):
        """
            input: [batch_size, input_size] tensor
            h: [batch_size, state_size] tensor
            C: [batch_size, state_size] tensor
            
            C: [batch_size, state_size] tensor
            h: [batch_size, state_size] tensor
            
            given inputs and last C & h, return new C & h
        """
        # dim: [batch_size, state_size]
        # forget gate
        f = self._gate(self.W_f, self.b_f, h, input)
        # input gate
        i = self._gate(self.W_i, self.b_i, h, input)
        # output gate
        o = self._gate(self.W_o, self.b_o, h, input)

        # [batch_size, state_size]
        C_hat = tf.tanh(
                tf.add(
                    tf.matmul(
                            tf.concat([h, input], 1),
                            self.W_C),
                    self.b_C)
                )
        C_new = tf.add(
                tf.multiply(f, C),
                tf.multiply(i, C_hat)
                )
        h_new = tf.multiply(o, tf.tanh(C_new))

        return C_new, h_new
    
    
    
    
    
    
    
    
    