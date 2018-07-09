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
            self.d_input_layer = tf.placeholder(tf.float32,
                shape = [self.batch_size, self.input_size],
                name = "d_input_layer")

        all_input_layer =  tf.concat([self.d_input_layer, generator_output], 0)
        last_layer = all_input_layer
        last_dim = self.input_size
        self.W_list = []
        self.b_list =[]
        with tf.variable_scope(self.scope_name):
            for idx in range(1, len(self.dim_list)):
                cur_dim = self.dim_list[idx]
                self.cur_weight = tf.get_variable("weigth" + str(idx), [last_dim, cur_dim],
                    initializer = tf.truncated_normal_initializer(stddev=0.02))
                self.cur_bias = tf.get_variable("bias" + str(idx), [cur_dim],
                    initializer=tf.constant_initializer(0.0))
                cur_act = self.activation_list[idx - 1]
                print(cur_act)
                last_layer = cur_act(tf.add(tf.matmul(last_layer, self.cur_weight), self.cur_bias))
                last_dim = cur_dim
                self.W_list.append(self.cur_weight)
                self.b_list.append(self.cur_bias)

        self.output_layer = last_layer
        self.d_output = tf.slice(self.output_layer, [0, 0], [self.batch_size, -1], name = None)
        self.g_output = tf.slice(self.output_layer, [self.batch_size, 0], [-1, -1], name = None)
        #print(self.output_layer.shape)
        #print(self.d_output.shape)
        #print(self.g_output.shape)
        return self.d_output, self.g_output
        
    def get_scope_name(self):
        return self.scope_name

    def get_var_list(self):
        return self.W_list + self.b_list

    def get_var_list_by_collection(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope_name)

    def get_input_layer_tensor(self):
        return self.d_input_layer


        
        
        
        
        
        
