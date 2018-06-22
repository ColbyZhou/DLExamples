import tensorflow as tf
from tensorflow.models.tutorials.rnn.ptb import reader

"""
To use models of tensorflow, firstly,
clone the repo `https://github.com/tensorflow/models`, and extrat to your intallation directory.
In my machine, it's `D:\software_install\Continuum\Anaconda3\Lib\site-packages\tensorflow` 

For `tensorflow.models.tutorials.rnn.ptb`, you should
remove 'import reader' & 'import util' in __init__.py,
Or, you will get an import error.
"""

"""
    Download data from 
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
"""


raw_data = reader.ptb_raw_data('ptb_data')
train_data, valid_data, test_data, voc_size = raw_data

batch_size = 200
num_steps = 5

def get_data(n):
    for i in range(n):
         yield reader.ptb_producer(train_data, batch_size, num_steps)

def get_raw_data(path):
    
    return []

def gen_data(raw_inputs, raw_labels, batch_size, num_steps_k1, num_steps_k2):
    
    batch_len = len(raw_inputs) / batch_size
    # inputs: `self.num_steps_k2` tensors with shape: [batch_size, embed_dim]
    
    batched_inputs = []
    batched_labels = []
    for i in range(batch_len):
        s = i * batch_size
        e = (i + 1) * batch_size
        batched_inputs[i] = raw_inputs[s:e]
        batched_labels[i] = raw_labels[s:e]
    
    size = batch_len / num_steps_k1

    for i in range(size):
        end_idx = (i + 1) * num_steps_k1
        start_idx = max(0, end_idx - num_steps_k2)
        
        input_list = batched_inputs[: start_idx : end_idx]
        label_list = batched_labels[: start_idx : end_idx]
        
        yield (input_list, label_list)

