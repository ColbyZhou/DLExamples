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
        
    def __init__(self, batch_size, dim_list, activation_list, learning_rate = 0.1,
        scope_name = None, extra_tensor_input = None):
        """
            define a MLP
            dim_list: dim list including input_size
            activation_list: activation function list, exlcuding input layer
            extra_tensor_input: extra tensor input for GANs
        """
        assert len(dim_list) > 1
        assert len(activation_list) == len(dim_list) - 1
        # get a uniq name for each object
        if scope_name is None:
            self.scope_name = util.get_uniq_object_name(self.__class__.__name__,
                                MultiLayerPerceptron_PER_GRAPH_NAME_UID_DICT)
        else:
            self.scope_name = scope_name

        
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
        # build network
        if extra_tensor_input is None:
            last_layer = self.input_layer
            self.has_extra_tensor_input = False
        else: # add extra input
            assert len(extra_tensor_input.shape) > 1
            assert self.input_layer.shape[1] == extra_tensor_input.shape[1]
            # last_layer dim: [self.batch_size + self.extra_input_size, self.input_size]
            last_layer = tf.concat([self.input_layer, extra_tensor_input], 0)
            self.extra_input_size = extra_tensor_input.shape[0]
            self.has_extra_tensor_input = True
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
                #print(self.cur_weight.shape)
                #print(self.cur_weight.name)
                #print(self.cur_bias.shape)
                #print(self.cur_bias.name)
        self.output_layer = last_layer

        # train
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    
    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
    
    def get_scope_name(self):
        return self.scope_name

    def get_var_list(self):
        return self.W_list + self.b_list

    def get_var_list_by_collection(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope_name)

    def get_input_layer_tensor(self):
        return self.input_layer

    def get_output_layer_tensor(self):
        return self.output_layer

    def get_label_placeholder(self):
        return self.label_placeholder

    def get_labels_tensor(self):
        return self.labels

    def creat_label_placeholder(self):
        with tf.variable_scope(self.scope_name):
            self.label_placeholder = tf.placeholder(tf.float32,
                            shape = [self.batch_size, self.output_size],
                            name = "label_placeholder")

    def get_all_labels(self, extra_data_label):
        self.labels = self.label_placeholder
        # concat extra data label for extra input
        if self.has_extra_tensor_input:
            assert extra_data_label != None
            assert len(extra_data_label.shape) > 1
            assert self.extra_input_size == extra_data_label.shape[0]
            assert self.output_size == extra_data_label.shape[1]
            # labels dim: [self.batch_size + self.extra_input_size, self.output_size]
            self.labels = tf.concat([self.label_placeholder, extra_data_label], 0)
        return self.labels

    def get_square_loss(self, extra_data_label = None):
        # get loss
        self.labels = self.get_all_labels(extra_data_label)
        self.loss = tf.square(self.labels - self.output_layer)
        self.total_loss = tf.reduce_mean(self.loss)
        return self.total_loss
    
    def get_binary_cross_entropy_loss(self, extra_data_label = None):
        self.labels = self.get_all_labels(extra_data_label)
        self.loss = -1 * (self.labels * tf.log(self.output_layer) + (1 - self.labels) * tf.log(1 - self.output_layer))
        self.total_loss = tf.reduce_mean(self.loss)
        return self.total_loss
    
    def get_softmax_cross_entropy_loss(self, extra_data_label = None):

        self.labels = self.get_all_labels(extra_data_label)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
                logits = self.output_layer, labels = self.labels)
        self.total_loss = tf.reduce_mean(self.loss)
        return self.total_loss

    def optimize_loss(self, sess, loss):
        self.step = self.optimizer.minimize(loss)
        self.initialize(sess)

    def forward(self, sess, input):
        output_layer = sess.run([self.output_layer],
                                     feed_dict = {self.input_layer:input})
        return output_layer
        
    def train_step(self, sess, input, label):
        """
            input: np.ndarray with shape [batch_size, input_size], where input_size is dim_list[0])
            label: np.ndarray with shape [batch_size, output_size], where output_size is dim_list[-1]
        """
        
        cur_loss, W_list, b_list, output_layer, _ = sess.run(
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
    
    tmp_input = np.random.choice(100, size = (batch_size, input_size))
    tmp_label = np.random.choice([0, 1], size = (batch_size, output_size))
    
    with tf.Session() as sess:
        MLP.optimize_loss(sess,  MLP.get_square_loss())
        tmp_output = MLP.forward(sess, tmp_input)
        print(tmp_output)
        
        loss = MLP.train_step(sess, tmp_input, tmp_label)
        print(loss)
    
if __name__ == "__main__":
    demo()
    