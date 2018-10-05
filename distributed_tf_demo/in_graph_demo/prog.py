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

if len(sys.argv) < 3:
    print "argv err"
    sys.exit(1)

job_name = sys.argv[1]
task_index = int(sys.argv[2])

ps_hosts = ['10.58.53.16:2221']
worker_hosts = ['10.58.53.16:2222', '10.58.53.16:2223']

cluster_spec = tf.train.ClusterSpec({"worker": worker_hosts, "ps": ps_hosts})

server = tf.train.Server(cluster_spec, job_name = job_name, task_index = task_index)

server.join()
