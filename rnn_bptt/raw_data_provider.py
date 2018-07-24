# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 17:39:26 2018

@author: zhouqiang02
"""

import tensorflow as tf
import numpy as np
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


def get_all_raw_data():
    raw_data = reader.ptb_raw_data('ptb_data')
    return raw_data

def get_label_by_data(data, end_id):
    label_data = data[0:]
    label_data.append(end_id)
    return label_data