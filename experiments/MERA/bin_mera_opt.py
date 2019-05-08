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

NUM_THREADS = 4
import os
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

import tensorflow as tf
import copy
import numpy as np
import time
import pickle
import tensornetwork.ncon_interface as ncon
import binary_mera_lib as bml
import binary_mera as bm
import misc_mera
from sys import stdout
import datetime
config = tf.ConfigProto()
config.intra_op_parallelism_threads = NUM_THREADS
config.inter_op_parallelism_threads = 1
tf.enable_eager_execution(config)
tf.enable_v2_behavior()

def optimize_binary_mera(use_gpu=False):
    fname = 'binary_mera_optimization'
    if not os.path.exists(fname):
        os.mkdir(fname)
    os.chdir(fname)
    rootdir = os.getcwd()

    DEVICES = tf.contrib.eager.list_devices()
    print("Available devices:")
    for i, device in enumerate(DEVICES):
        print("%d) %s" % (i, device))
    CPU = '/device:CPU:0'
    GPU = '/job:localhost/replica:0/task:0/device:GPU:0'
    if use_gpu:
        specified_device_type = GPU
        name = 'GPU'
    else:
        specified_device_type = CPU
        name = 'CPU'

    num_trans_layers = 3
    chis = [4] * num_trans_layers + [6, 8, 10, 12, 14, 16]
    numiters = [1000, 1000, 500, 500, 200, 200, 200, 200
               ] + [500, 400, 300, 200, 200, 800]
    noises = [1E-6] * num_trans_layers + [1E-7, 1E-8, 1E-9, 1E-10, 1E-11, 0.0]
    opt_all_layers = [True] * len(chis)
    embeddings = ['a'] * num_trans_layers + ['p'
                                            ] * (len(chis) - num_trans_layers)
    dtype = tf.float64
    nsteps_ss = 12
    filename = str(datetime.date.today()
                  ) + '_bin_mera_opt_Nthreads{0}_chimax{1}_numtrans{2}'.format(
                      NUM_THREADS, max(chis), num_trans_layers)
    with tf.device(device):
        wC, uC, _, _ = bm.run_binary_mera_optimization_TFI(
            chis=chis,
            niters=numiters,
            embeddings=embeddings,
            dtype=dtype,
            verbose=1,
            nsteps_steady_state=nsteps_ss,
            numpy_update=True,
            opt_u_after=10,
            noises=noises,
            opt_all_layers=opt_all_layers,
            wC=wC,
            uC=uC
            filename=filename)
    os.chdir(rootdir)

def load_and_optimize_binary_mera(filename, use_gpu=False):
    fname = 'binary_mera_optimization'
    if not os.path.exists(fname):
        os.mkdir(fname)
    os.chdir(fname)
    rootdir = os.getcwd()


    DEVICES = tf.contrib.eager.list_devices()
    print("Available devices:")
    for i, device in enumerate(DEVICES):
        print("%d) %s" % (i, device))
    CPU = '/device:CPU:0'
    GPU = '/job:localhost/replica:0/task:0/device:GPU:0'
    if use_gpu:
        specified_device_type = GPU
        name = 'GPU'
    else:
        specified_device_type = CPU
        name = 'CPU'
    with open(filename, 'rb') as f:
        wC, uC = pickle.load(f)
        
    num_trans_layers = 2 #add 2 new layers
    chis = [wC[-1].shape[2]] * num_trans_layers
    numiters = [100] * num_trans_layers
    noises = [0.0] * num_trans_layers
    opt_all_layers = [True] * num_trans_layers
    embeddings = ['a'] * num_trans_layers 
    dtype = tf.float64
    nsteps_ss = 14
    filename = str(datetime.date.today()
                  ) + 'resumed_bin_mera_opt_Nthreads{0}_chimax{1}_numtrans{2}_nss{3}'.format(
                      NUM_THREADS, max(chis), num_trans_layers + len(wC), nsteps_ss)
    with tf.device(device):
        wC, uC, _, _ = bm.run_binary_mera_optimization_TFI(
            chis=chis,
            niters=numiters,
            embeddings=embeddings,
            dtype=dtype,
            verbose=1,
            nsteps_steady_state=nsteps_ss,
            numpy_update=True,
            opt_u_after=0,
            noises=noises,
            opt_all_layers=opt_all_layers,
            wC=wC,
            uC=uC
            filename=filename)
    os.chdir(rootdir)
    
if __name__ == "__main__":
    #optimize_binary_mera(use_gpu=False)
    load_and_optimize_binary_mera(filename, use_gpu=False)
