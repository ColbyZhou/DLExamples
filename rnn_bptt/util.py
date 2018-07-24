# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:04:30 2018

@author: zhouqiang02
"""
import re
import collections
import tensorflow as tf

# Borrowed from tensorlow/python/layers/base.py _to_snake_case(1446)
def to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure

# Borrowed from tensorlow/python/layers/base.py 
# _init_set_name(158) => _make_unique_name(450) => _unique_layer_name(1545)
def get_uniq_object_name(class_name, PER_GRAPH_NAME_UID_DICT):
    """
        create a uniq name for each object
    """
    # get base_name from class name
    base_name = to_snake_case(class_name)
    
    # get name_uid_map with graph as key
    graph = tf.get_default_graph()
    name_uid_map = PER_GRAPH_NAME_UID_DICT.get(graph, None)
    if name_uid_map is None:
        name_uid_map = collections.defaultdict(int)
        PER_GRAPH_NAME_UID_DICT[graph] = name_uid_map
    
    # get uniq id for each object
    name_uid_map[base_name] += 1
    uniq_name = base_name + '_' + str(name_uid_map[base_name])
    return uniq_name

