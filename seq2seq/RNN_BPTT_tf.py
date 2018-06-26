# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 17:14:16 2018

@author: zhouqiang02
"""

import numpy as np
import tensorflow as tf
import raw_data_provider
import data_transfer

num_steps_k1 = 5
num_steps_k2 = 10

batch_size = 200
state_size = 50

#embed_dim = 10

learning_rate = 0.01
epoch_num = 1

class MyRNNCell:
    """
        RNN by my own
    """
    def __init__(self, state_size, embed_dim):
        """
        state_size: hidden layer size
        embed_dim: input_size
        """
        self.state_size = state_size
        self.embed_dim = embed_dim
        self.W = tf.get_variable(
                name = 'W',
                shape = [self.state_size + self.embed_dim, self.state_size],
                #initializer=tf.constant_initializer(0.1)
                )
        self.b = tf.get_variable(
                name = 'b',
                shape = [self.state_size],
                initializer=tf.constant_initializer(0.1))
            
    def one_time_step(self, input, state):
        """
            input: [batch_size, embed_dim] tensor
            state: [batch_size, state_size] tensor
            output: [batch_size, state_size] tensor
            
            given inputs and last state, return output
            h(t) = tanh(W*h(t-1) + U*X + b)
        """
        
        output = tf.tanh(
                tf.matmul(
                        tf.concat([input, state], 1),
                        self.W 
                        )
                + self.b
                )
        return output

class RNN_Truncated_BPTT:
    """
    RNN use Truncated BPTT for training
    """
    
    def __init__(self, num_steps_k1, num_steps_k2, batch_size, state_size,
                 embed_dim, learning_rate):
        self.num_steps_k1 = num_steps_k1
        self.num_steps_k2 = num_steps_k2
        self.batch_size = batch_size
        self.state_size = state_size
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.init_state = tf.zeros([self.batch_size, self.state_size])
        
        self.my_rnn_cell = MyRNNCell(self.state_size, self.embed_dim)
        self.tf_rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units = self.state_size)

        self.o_W = tf.get_variable(
                name = 'o_W',
                shape = [self.state_size, self.embed_dim],
                #initializer=tf.constant_initializer(0.1)
                )
        self.o_b = tf.get_variable(
                name = 'o_b',
                shape = [self.embed_dim],
                initializer=tf.constant_initializer(0.1))
        
        self.sess = tf.Session()
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        #self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    
    def build_with_my_rnn(self, input_list):
        """
        build RNN
        input_list: list of `num_steps_k2` tensors with shape: [batch_size, embed_dim]
        output_list: list of `num_steps_k2` tensors with shape: [batch_size, state_size]
        final_state: last state
        """
        
        self.output_list = []
        state = self.init_state
        for input in input_list:
            state = self.my_rnn_cell.one_time_step(input, state)
            self.output_list.append(state)
        self.final_state = self.output_list[-1]

    def build_with_tf_rnn(self, input_list):
        """
        build RNN
        input_list: list of `num_steps_k2` tensors with shape: [batch_size, embed_dim]
        output_list: list of `num_steps_k2` tensors with shape: [batch_size, state_size]
        final_state: last state
        """
        # use high-level api BasicRNNCell
        self.output_list, self.final_state = tf.contrib.rnn.static_rnn(
                cell=self.tf_rnn_cell, inputs=input_list, initial_state=self.init_state)

    def build_output_layer(self, lable_list):
        """
        get loss of trainning exmples
        output_list: list of `num_steps_k2` tensors with shape: [batch_size, state_size]
        lable_list: same shape with output_list
        """
        losses = []
        preds = []
        # output_list length: num_steps_k2
        for idx, output in enumerate(self.output_list):
            # output shape: [batch_size, state_size]
            logit = tf.matmul(
                output,
                self.o_W,
            ) + self.o_b

            # logit & pred shape: [batch_size, embed_size]
            pred = tf.nn.softmax(logit)
            preds.append(pred)
            label= lable_list[idx]
            loss = tf.nn.softmax_cross_entropy_with_logits(labels = label,
                                                          logits = logit)
            losses.append(loss)
        
        # losses : num_steps_k2 rows, each's dim is [batch_size]
        
        total_loss = tf.reduce_mean(losses, axis = None)
        return total_loss, preds
    
    def build_graph(self):
        """
        build computation graph
        """
        if self.num_steps_k2 < self.num_steps_k1:
            print("num_steps_k2 < num_steps_k1, exit")
            return

        self.input_list_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.num_steps_k2])
        self.label_list_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.num_steps_k2])
        
        # dim [batch_size, num_steps_k2, embed_dim]
        input_list = tf.one_hot(self.input_list_placeholder, self.embed_dim)
        # array of num_steps_k2 tensors, with dim [batch_size, embed_dim]
        input_list = tf.unstack(input_list, axis = 1)
        
        label_list = tf.one_hot(self.label_list_placeholder, self.embed_dim)
        label_list = tf.unstack(label_list, axis = 1)

        self.build_with_my_rnn(input_list)
        
        # output_list[k1 - 1] as last_state
        self.last_state = self.output_list[num_steps_k1 - 1]
        
        # output layer
        self.total_loss, self.preds = self.build_output_layer(label_list)
        self.training_step = self.optimizer.minimize(self.total_loss)
        
        # initialize
        self.sess.run(tf.global_variables_initializer())

    def truncated_BPTT(self, raw_inputs, raw_labels, epoch_num):
        """
        Truncated Backpropagation Through Time (Truncated BPTT)
        BPTT(k2, k1) means that processes the sequence one timestep at a time,
        and every k1 timesteps, it runs BPTT for k2 timesteps, where
        (1 <= k1 <= k2)
        
        Algorithm:
        1: for t from 1 to T do
        2:      Run the RNN for one step, computing h_t and z_t
        3:      if t divides k1 then
        4:          Run BPTT from t down to (t - k2)
        5:      end if
        6: end for
    
        from <<TRAINING RECURRENT NEURAL NETWORKS>> (2.8.6) by
        Ilya Sutskever (2013)
        
        According to
        <<An Efficien Gradient-Based Algorithm for On-Line Training of
        Recurrent Network Trajectories>> by Williams and Peng (1990),
        BPTT(2n, n) is essentially identical to that of BPTT(n) [or BPTT(n, 1)]
        
        Tensorflow-style truncated backpropagation uses k1 = k2 (= num_steps)
       
        Reference:
        https://r2rt.com/styles-of-truncated-backpropagation.html
        https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
        
        """
        
        if self.num_steps_k2 < self.num_steps_k1:
            print("num_steps_k2 < num_steps_k1, exit")
            return

        train_state = np.zeros([batch_size, state_size])
        for num in range(epoch_num):
            # all_data: a list of tuple (input_list, label_list), 
            # including all data
            all_data = data_transfer.gen_data(raw_inputs, raw_labels,
                            self.embed_dim, self.batch_size,
                            self.num_steps_k1, self.num_steps_k2)
            print("epoc: " + str(num))
            for data_idx, (input_list, label_list) in enumerate(all_data):
                # get current input_list & label_list
                # input_list: `self.num_steps_k2` tensors 
                # with shape: [batch_size, embed_dim]
                # (input_list size is `self.num_steps_k1` for first step)
                
                feed_dict = dict()
                feed_dict = {
                             self.init_state:train_state,
                             self.input_list_placeholder:input_list,
                             self.label_list_placeholder:label_list
                             }
                
                train_state, output_list, train_loss, _ = self.sess.run([self.last_state,
                                                           self.output_list,
                                                           self.total_loss,
                                                           self.training_step],
                                                           feed_dict = feed_dict)
                if data_idx % 10 == 0:
                    print(train_state)
                    print("train_loss at epoc" + str(num) + " data_idx: "
                          + str(data_idx) + ' : ' + str(train_loss))

def main():
    
    # list of ids
    train_data, valid_data, test_data, voc_size = raw_data_provider.get_all_raw_data()
    
    raw_inputs = train_data
    
    embed_dim = voc_size + 1
    raw_labels = raw_data_provider.get_label_by_data(raw_inputs, voc_size)
    
    rnn = RNN_Truncated_BPTT(num_steps_k1, num_steps_k2, batch_size, state_size,
                         embed_dim, learning_rate)
    
    rnn.build_graph()
    
    rnn.truncated_BPTT(raw_inputs, raw_labels, epoch_num)
    
    print("done")

if __name__ == '__main__':
    main()