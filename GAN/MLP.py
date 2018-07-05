# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:52:38 2018

@author: zhouqiang02
"""
import numpy as np
import tensorflow as tf
import util
import weakref

MultiLayerPerceptron_PER_GRAPH_NAME_UID_DICT = weakref.WeakKeyDictionary()

class MultiLayerPerceptron:
        
    def __init__(self, batch_size, dim_list, activation_list, learning_rate = 0.1):
        """
            define a MLP
            dim_list: dim list including input_size
            activation_list: activation function list, exlcuding input layer
        """
        assert len(dim_list) > 1
        assert len(activation_list) == len(dim_list) - 1
        
        self.scope_name = util.get_uniq_object_name(self.__class__.__name__,
                            MultiLayerPerceptron_PER_GRAPH_NAME_UID_DICT)
        
        self.batch_size = batch_size
        self.dim_list = dim_list
        self.activation_list = activation_list
        self.learning_rate = learning_rate
        
        self.input_size = self.dim_list[0]
        self.output_size = self.dim_list[-1]
        with tf.variable_scope(self.scope_name):
            self.input_layer = tf.placeholder(tf.float32,
                                    shape = [self.batch_size, self.input_size],
                                    name = "input_layer")
            self.label_placeholder = tf.placeholder(tf.float32,
                                    shape = [self.batch_size, self.output_size],
                                    name = "label_placeholder")
        # build network
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
                last_layer = tf.add(cur_act(tf.matmul(last_layer, self.cur_weight)), self.cur_bias)
                last_dim = cur_dim
                self.W_list.append(self.cur_weight)
                self.b_list.append(self.cur_bias)
                print(self.cur_weight.shape)
                print(self.cur_weight.name)
                print(self.cur_bias.shape)
                print(self.cur_bias.name)
        self.output_layer = last_layer
        
        # train
        self.sess = tf.Session()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    
    def initialize(self):
        # initialization
        self.sess.run(tf.global_variables_initializer())
    
    def opt_by_square_loss(self):
        # get loss
        self.loss = tf.square(self.label_placeholder - self.output_layer)
        self.total_loss = tf.reduce_mean(self.loss)
        self.step = self.optimizer.minimize(self.total_loss)
        self.initialize()
    
    def opt_by_softmax_cross_entropy_loss(self):
        
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
                features = self.output_layer, labels = self.label_placeholder)
        self.total_loss = tf.reduce_mean(self.loss)
        self.step = self.optimizer.minimize(self.total_loss)
        self.initialize()

    def forward(self, input):
        output_layer = self.sess.run([self.output_layer],
                                     feed_dict = {self.input_layer:input})
        return output_layer
    
    def get_lable_placeholder(self):
        return self.label_placeholder
        
    def train_step(self, input, label):
        """
            input: np.ndarray with shape [batch_size, input_size], where input_size is dim_list[0])
            label: np.ndarray with shape [batch_size, output_size], where output_size is dim_list[-1]
        """
        
        cur_loss, W_list, b_list, output_layer, _ = self.sess.run(
        [self.total_loss, self.W_list, self.b_list, self.output_layer, self.step],
        feed_dict = {self.input_layer:input, self.label_placeholder:label})

        #for layer in layer_list:
        #    print(layer.shape)
        #print(output_layer.shape)
        return cur_loss, output_layer


def demo():
    batch_size = 100
    input_size = 64
    output_size = 1
    
    MLP = MultiLayerPerceptron(batch_size, [input_size, 128, 128, output_size],
                                           [tf.tanh, tf.tanh, tf.sigmoid])
    MLP.opt_by_square_loss()
    
    tmp_input = np.random.choice(100, size = (batch_size, input_size))
    tmp_label = np.random.choice([0, 1], size = (batch_size, output_size))
    
    tmp_output = MLP.forward(tmp_input)
    #print(tmp_output)
    
    loss = MLP.train_step(tmp_input, tmp_label)
    print(loss)
    
if __name__ == "__main__":
    demo()
    