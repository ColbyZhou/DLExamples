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

import tensorflow as tf

if len(sys.argv) < 2:
    print "usage: python " + sys.argv[0] + " task_index"
    sys.exit(1)

task_index = int(sys.argv[1])

ps_hosts = ['10.58.53.16:2221']
worker_hosts = ['10.58.53.16:2222', '10.58.53.16:2223']

cluster = tf.train.ClusterSpec({"worker": worker_hosts, "ps": ps_hosts})

server = tf.train.Server(cluster, job_name = "ps", task_index = task_index)

server.join()
