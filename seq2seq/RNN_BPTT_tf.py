# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 17:14:16 2018

@author: zhouqiang02
"""


import tensorflow as tf
import raw_data_provider
import data_transfer

num_steps_k1 = 5
num_steps_k2 = 10
batch_size = 200
state_size = 10
embed_dim = 10
learning_rate = 0.01
epoch_num = 1

init_state = tf.zeros([batch_size, state_size])

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
                initializer=tf.constant_initializer(0.0))
        self.b = tf.get_variable(
                name = 'b',
                shape = [self.state_size],
                initializer=tf.constant_initializer(0.0))
    
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
                 embed_dim, learning_rate, init_state):
        self.num_steps_k1 = num_steps_k1
        self.num_steps_k2 = num_steps_k2
        self.batch_size = batch_size
        self.state_size = state_size
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.init_state = init_state
    
    def construct_data(self):
        """
            get placeholder for data
        """
        
        # need go self.num_steps_k2 steps, ever self.num_steps_k1 step
        self.x = tf.placeholder(tf.int32, [self.batch_size, self.num_steps_k2],
                                name='inputs')
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.num_steps_k2],
                                name='labels')
        
        # [batch_size, num_steps_k2, embed_dim]
        self.x_one_hot = tf.one_hot(self.x, self.embed_dim)
        
        # array of [batch_size, embed_dim] with length `num_steps_k2`
        self.inputs_for_one_step = tf.unstack(self.x_one_hot, axis = 1)
    
    def go_with_my_rnn(self, input_list, init_state):
        """
        build RNN
        input_list: list of `num_steps` tensors with shape: [batch_size, embed_dim]
        output_list: list of `num_steps` tensors with shape: [batch_size, state_size]
        final_state: last state
        """
        cell = MyRNNCell(self.state_size, self.embed_dim)
        
        output_list = []
        state = init_state
        for input in input_list:
            state = cell.one_time_step(input, state)
            output_list.append(state)
        final_state = output_list[-1]
        
        return output_list, final_state
    
    def go_with_tf_rnn(self, input_list, init_state):
        """
        build RNN
        input_list: list of `num_steps` tensors with shape: [batch_size, embed_dim]
        output_list: list of `num_steps` tensors with shape: [batch_size, state_size]
        final_state: last state
        """
        # use high-level api BasicRNNCell
        cell = tf.contrib.rnn.BasicRNNCell(num_units=self.state_size)
        output_list, final_state = tf.contrib.rnn.static_rnn(
                cell=cell, inputs=input_list, initial_state=init_state)
        
        return output_list, final_state
                
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
        init_state = self.init_state
        for num in range(epoch_num):
            # all_data: a list of tuple (input_list, label_list), 
            # including all data
            all_data = data_transfer.gen_data(raw_inputs, raw_labels,
                            self.embed_dim, self.batch_size,
                            self.num_steps_k1, self.num_steps_k2)

            for data_idx, (input_list, label_list) in enumerate(all_data):
                # get current input_list & label_list
                # input_list: `self.num_steps_k2` tensors 
                # with shape: [batch_size, embed_dim]
                # (input_list size is `self.num_steps_k1` for first step)
                
                output_list, final_state = self.go_with_my_rnn(input_list, init_state)
                
                # output_list[k1 - 1] as init_state
                last_state_idx = min(len(output_list) - 1, num_steps_k1 - 1)
                init_state = output_list[last_state_idx]
                
                # output layer

def main():
    
    rnn = RNN_Truncated_BPTT(num_steps_k1, num_steps_k2, batch_size, state_size,
                             embed_dim, learning_rate, init_state)
    
    # list of ids
    data_path = ''
    raw_inputs, raw_labels = raw_data_provider.get_raw_data(data_path)
    
    rnn.truncated_BPTT(raw_inputs, raw_labels, epoch_num)
    
    print("done")
    
        
    
























    
        
        
        