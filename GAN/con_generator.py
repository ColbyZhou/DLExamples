# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:32:07 2018

@author: zhouqiang02
"""

import tensorflow as tf
import util
import weakref

ConGenerator_PER_GRAPH_NAME_UID_DICT = weakref.WeakKeyDictionary()

class ConGenerator:
    
    def __init__(self, batch_size, dim_list, activation_list, label_num,
                 label_embed_dim, scope_name = None):
        """
            define a Generator
            dim_list: dim list including input_size
            activation_list: activation function list, exlcuding input layer
        """     
        assert len(dim_list) > 1
        assert len(activation_list) == len(dim_list) - 1
        # get a uniq name for each object
        if scope_name is None:
            self.scope_name = util.get_uniq_object_name(self.__class__.__name__,
                                ConGenerator_PER_GRAPH_NAME_UID_DICT)
        else:
            self.scope_name = scope_name
        
        self.batch_size = batch_size
        self.dim_list = dim_list
        self.activation_list = activation_list
        self.label_num = label_num
        self.label_embed_dim = label_embed_dim
        
        self.input_size = self.dim_list[0]
        self.output_size = self.dim_list[-1]
    
    def build(self):
        """
            build Generator network 
        """

        with tf.variable_scope(self.scope_name):
            self.input_layer = tf.placeholder(tf.float32,
                shape = [self.batch_size, self.input_size],
                name = "input_layer")
            self.input_label = tf.placeholder(tf.int32, 
                shape = [self.batch_size],
                name = "input_label")
            self.label_embedings = tf.get_variable("label_embedings",
                                    [self.label_num, self.label_embed_dim])
            # [batch_size, label_embed_dim]
            self.input_label_embed = tf.nn.embedding_lookup(
                    self.label_embedings, self.input_label)

        # [batch_size, input_size + label_embed_dim]
        last_layer = tf.concat([self.input_layer, self.input_label_embed], 1)
        last_dim = self.input_size + self.label_embed_dim
        self.W_list = []
        self.b_list =[]
        with tf.variable_scope(self.scope_name):
            for idx in range(1, len(self.dim_list)):
                cur_dim = self.dim_list[idx]
                self.cur_weight = tf.get_variable("weigth" + str(idx), [last_dim, cur_dim],
                    #initializer = tf.truncated_normal_initializer(stddev=0.02)
                    )
                self.cur_bias = tf.get_variable("bias" + str(idx), [cur_dim],
                    #initializer=tf.constant_initializer(0.0)
                    )
                tmp_layer = tf.add(tf.matmul(last_layer, self.cur_weight), self.cur_bias)
                cur_act = self.activation_list[idx - 1]
                if cur_act != None:
                    last_layer = cur_act(tmp_layer)
                else:
                    last_layer = tmp_layer
                last_dim = cur_dim
                self.W_list.append(self.cur_weight)
                self.b_list.append(self.cur_bias)

        self.output_layer = last_layer
        
        return self.output_layer

    def get_scope_name(self):
        return self.scope_name

    def get_var_list(self):
        return self.W_list + self.b_list

    def get_var_list_by_collection(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope_name)

    def get_input_layer_tensor(self):
        return self.input_layer
    
    def get_input_label_tensor(self):
        return self.input_label
    