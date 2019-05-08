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

NUM_THREADS = 4
import os
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

import sys
sys.path.append('../../')

import tensorflow as tf
import copy
import numpy as np
import time
import pickle
import ncon_tn as ncon
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


if __name__ == "__main__":
    fname =  'binary_mera_optimization'            
    if not os.path.exists(fname):
        os.mkdir(fname)
    os.chdir(fname)
    rootdir = os.getcwd()

    use_gpu = False
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
    #chis = [4] * num_trans_layers + [6, 8, 10, 12, 14, 16]
    chis = [4] * num_trans_layers + [6, 6,6,6,6,6]    
    numiters = [1000, 1000, 500, 500, 200, 200, 200, 200] + [500, 400, 300, 200, 200, 800]
    noises = [1E-6] * num_trans_layers + [1E-7, 1E-8, 1E-9, 1E-10, 1E-11, 0.0]
    opt_all_layers = [True] * len(chis)
    embeddings = ['a'] * num_trans_layers + ['p'] * (len(chis) - num_trans_layers)
    dtype = tf.float64
    nsteps_ss = 8
    filename = str(datetime.date.today())+ '_bin_mera_opt_Nthreads{0}_chimax{1}_numtrans{2}'.format(NUM_THREADS, max(chis), num_trans_layers)
    with tf.device(device):
        wC, uC, _, _ = bm.run_binary_mera_optimization_TFI(chis=chis,
                                                           niters=numiters,
                                                           embeddings=embeddings,
                                                           dtype=dtype,
                                                           verbose=1,
                                                           nsteps_steady_state=nsteps_ss,
                                                           numpy_update=True,
                                                           opt_u_after=10,
                                                           noises=noises,
                                                           opt_all_layers=opt_all_layers,
                                                           filename=filename)
    os.chdir(rootdir)                        



            
