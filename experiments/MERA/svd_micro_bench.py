# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

NUM_THREADS = 1
import os
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

import tensorflow as tf
import copy
import datetime
import numpy as np
import time
import pickle
import tensornetwork as tn
import experiments.MERA.binary_mera_lib as bml
import experiments.MERA.misc_mera as misc_mera
from sys import stdout
config = tf.ConfigProto()
config.intra_op_parallelism_threads = NUM_THREADS
config.inter_op_parallelism_threads = 1
tf.enable_eager_execution(config)
tf.enable_v2_behavior()

if __name__ == "__main__":
    """
    """
    use_gpu = True  #use True when running on GPU
    #list available devices
    DEVICES = tf.contrib.eager.list_devices()
    print("Available devices:")
    for i, device in enumerate(DEVICES):
        print("%d) %s" % (i, device))
    CPU = '/device:CPU:0'
    GPU = '/job:localhost/replica:0/task:0/device:GPU:0'
    if use_gpu:
        specified_device_type = GPU
    else:
        specified_device_type = CPU
    D=16
    #tens = np.load('16uEnv16uEnv16uEnv16.npy')
    tens = np.load('2uEnv2uEnv2uEnv2.npy')    
    with tf.device(device):
        #tensor = tf.random_uniform(shape=[D,D,D,D], dtype=tf.float64,minval=-0.5,maxval=0.5)
        tensor = tf.convert_to_tensor(tens)
        for _ in range(100):
            t1 = time.time()
            #u,s,v = np.linalg.svd(tf.reshape(tensor,(D*D,D*D)), full_matrices=False)
            #tensor = -tf.reshape(tn.ncon([u,v],[[-1,1],[1,-2]]),(D,D,D,D))
            out = misc_mera.u_update_svd_numpy(tensor)
            print(time.time()-t1)
        
