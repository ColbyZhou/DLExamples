#!/usr/bin/python
# -*- coding:utf-8 -*-
# author: zhouqiang02
# breif:

import sys
import os
import urlparse
import urllib
import sys
import json
import os
import re
import urllib2

import numpy as np
import tensorflow as tf

if len(sys.argv) < 2:
    print "usage: python " + sys.argv[0] + " task_index"
    sys.exit(1)

task_index = int(sys.argv[1])

lr = 1e-2
total_run_num = 2000

ps_hosts = ['localhost:2221']
worker_hosts = ['localhost:2222', 'localhost:2223']

# 1. Create a cluster
cluster = tf.train.ClusterSpec({"worker": worker_hosts, "ps": ps_hosts})

# 2. Create a server for local task
server = tf.train.Server(cluster, job_name = "worker", task_index = task_index)

is_chief = (task_index == 0)
print "is_chief: " + str(is_chief)

# ******* (Application Specific) Simulate Train Data *******
train_data_X = np.linspace(-1, 1, 100)
train_data_Y = 2 * train_data_X + 10 + np.random.randn(*train_data_X.shape) * 0.1

data_idx = 0
epoc_num = 0
def get_data_batch():

    global data_idx
    global epoc_num
    while True:
        if data_idx == len(train_data_X) - 1:
            epoc_num += 1
        data_idx = data_idx % len(train_data_X)
        cur_data = (train_data_X[data_idx], train_data_Y[data_idx])
        data_idx += 1
        yield cur_data

data_generator = get_data_batch()

# 3. Set replica devices
# def replica_device_setter(ps_tasks=0, ps_device="/job:ps",
#                          worker_device="/job:worker", merge_devices=True,
#                          cluster=None, ps_ops=None, ps_strategy=None):
with tf.device(tf.train.replica_device_setter(
    worker_device = "/job:worker/task:%d" % task_index,
    cluster = cluster)):

    # 4. ******* (Application Specific) build model *******
    W = tf.get_variable('W', [], dtype = tf.float32, initializer=tf.zeros_initializer)
    b = tf.get_variable('b', [], dtype = tf.float32, initializer=tf.zeros_initializer)

    X = tf.placeholder(dtype = tf.float32, shape = [], name = "input_x")
    Y = tf.placeholder(dtype = tf.float32, shape = [], name = "label_y")

    loss = tf.square(tf.subtract(tf.add(tf.multiply(W, X), b), Y))

    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    # 5. global_step
    global_step = tf.train.get_or_create_global_step()

    # 6. ******* (Method Specific) sync optimizer *******
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step = global_step)

    # 7. define stop steps. Will Stop after running given steps
    hooks = [tf.train.StopAtStepHook(last_step = total_run_num)]

    # 8. Create a monitor session
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(
            master = server.target,
            is_chief = is_chief,
            checkpoint_dir = './chk_point_asyn/',
            hooks = hooks
            ) as mon_sess:
        # 9. Iterate total_run_num times (num of mon_sess.run)
        _W = 0
        _b = 0
        local_step = 0
        while not mon_sess.should_stop():
            # 10. ******* (Application Specific) Training Process *******
            x, y = data_generator.next()
            _, cur_loss, cur_step, _W, _b = mon_sess.run([train_op, loss, global_step, W, b], feed_dict = {X: x, Y: y})
            local_step += 1
            print "loss: " + str(cur_loss) + ', step: ' + str(cur_step) + ' local_step: ' + str(local_step) + ', _W: ' + str(_W) + ', _b: ' + str(_b)

        print "****************************result:**********************" + str(_W) + ', ' + str(_b)
