# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 19:52:23 2018

@author: zhouqiang02
"""

class DataLoader(object):
    """DataLoader"""
    def __init__(self, file_path):
        super(DataLoader, self).__init__()
        self.file_path = file_path

    def read_vocab(self, voc_path):
        """
            read vocabulary
        """
        self.word_to_id = dict()
        self.id_to_word = dict()
        with open(voc_path, 'r') as file:
            for idx, line in enumerate(file):
                fields = line.strip().split('\t')
                _word = fields[0]
                _id = int(fields[1])
                self.word_to_id[_word] = _id
                self.id_to_word[_id] = _word

    def read_data(self):
        """
            read data
            line format: input \t output
            with, input(output) is a sequence of word id, seperated by space
        """
        self.input_data = []
        self.output_data = []
        with open(self.file_path, 'r') as file:
            for idx, line in enumerate(file):
                fields = line.strip().split('\t')
                input_seq = [int(x) for x in fields[0].split(' ')]
                output_seq = [int(x) for x in fields[1].split(' ')]
                self.input_data.append(input_seq)
                self.output_data.append(output_seq)

        self._num_examples = len(self.input_data)
        self._begin = 0
        self._epochs = 0

    def next_batch(self, batch_size):
        """
            get next batch data
        """
        start = self._begin
        batch_input = []
        batch_output = []
        if start + batch_size > self._num_examples:
            self._epochs += 1
            rest_num = self._num_examples - start
            self._begin = batch_size - rest_num
            batch_input = self.input_data[start:] + self.input_data[0:self._begin]
            batch_output = self.output_data[start:] + self.output_data[0:self._begin]
        else:
            end = start + batch_size
            batch_input = self.input_data[start:end]
            batch_output = self.output_data[start:end]
            self._begin = end
        return (batch_input, batch_output)

    def padding_batch(self, batch, seq_len):
        """
            pad the batch
        """
        if '<PAD>' not in self.word_to_id:
            raise Exception("<PAD> not in vocabulary")
        pad = self.word_to_id['<PAD>']
        pad_batch = []
        seq_lengths = []
        for seq in batch:
            new_seq = seq + (seq_len - len(seq)) * [pad]
            pad_batch.append(new_seq)
            seq_lengths.append(len(seq))

        return (pad_batch, seq_lengths)

    def get_word_to_id(self):
        return self.word_to_id

    def get_id_to_word(self):
        return self.id_to_word

    def get_num_examples(self):
        return self._num_examples

    def get_epochs(self):
        return self._epochs

def test():

    data_loader = DataLoader("temp.txt")
    data_loader.read_data()

    batch_size = 6
    for i in range(5):
        batch = data_loader.next_batch(batch_size)
        print(batch)
        print("==============")

if __name__ == '__main__':
    test()