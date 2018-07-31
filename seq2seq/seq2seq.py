# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 18:01:04 2018

@author: zhouqiang02
"""

from dataloader import DataLoader 

def train_seq2seq(train_path, voc_path):

    data_loader = DataLoader(train_path)
    data_loader.read_data()
    dataloader.read_vocab(voc_path)

    word_to_id = dataloader.get_word_to_id()
    id_to_word = dataloader.get_id_to_word()

    epoc_num = 1
    batch_size = 64

    batch_num = int(data_loader.get_num_examples() / batch_size)

    for i in range(epoc_num):
        for j in range(batch_num):
            batch = data_loader.next_batch()
            batch_input, batch_output = batch
            pad_batch_input, pad_batch_input_lengths = data_loader.padding_batch(batch_input, seq_len)
            pad_batch_output, pad_batch_output_lengths = data_loader.padding_batch(batch_output, seq_len)

if __name__ == '__main__':
    seq2seq()