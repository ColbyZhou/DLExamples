# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 19:31:19 2018

@author: zhouqiang02
"""

import tensorflow as tf
import util
import weakref

Discriminator_PER_GRAPH_NAME_UID_DICT = weakref.WeakKeyDictionary()

class Discriminator:
    
    def __init__(self, batch_size, dim_list, activation_list, scope_name = None):
        """
            define a Discriminator
            dim_list: dim list including input_size
            activation_list: activation function list, exlcuding input layer
        """
        assert len(dim_list) > 1
        assert len(activation_list) == len(dim_list) - 1
        # get a uniq name for each object
        if scope_name is None:
            self.scope_name = util.get_uniq_object_name(self.__class__.__name__,
                                Discriminator_PER_GRAPH_NAME_UID_DICT)
        else:
            self.scope_name = scope_name

        self.batch_size = batch_size
        self.dim_list = dim_list
        self.activation_list = activation_list
        
        self.input_size = self.dim_list[0]
        self.output_size = self.dim_list[-1]
    
    def build(self, generator_output):
        """
            build Discriminator network 
        """
        
        with tf.variable_scope(self.scope_name):
            self.input_layer = tf.placeholder(tf.float32,
                shape = [self.batch_size, self.input_size],
                name = "input_layer")

        last_layer = self.input_layer
        last_dim = self.input_size
        self.W_list = []
        self.b_list =[]
        with tf.variable_scope(self.scope_name):
            for idx in range(1, len(self.dim_list)):
                cur_dim = self.dim_list[idx]
                self.cur_weight = tf.get_variable("weigth" + str(idx), [last_dim, cur_dim])                
                self.cur_bias = tf.get_variable("bias" + str(idx), [cur_dim])
                cur_act = self.activation_list[idx - 1]
                last_layer = cur_act(tf.add(tf.matmul(last_layer, self.cur_weight), self.cur_bias))
                last_dim = cur_dim
                self.W_list.append(self.cur_weight)
                self.b_list.append(self.cur_bias)

        self.output_layer = last_layer
        
        
        
        
        
        
        
        
