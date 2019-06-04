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
import experiments.MERA.binary_mera_lib as bml
import experiments.MERA.binary_mera as bm
import experiments.MERA.misc_mera as misc_mera
from sys import stdout
import datetime
config = tf.ConfigProto()
config.intra_op_parallelism_threads = NUM_THREADS
config.inter_op_parallelism_threads = 1
tf.enable_eager_execution(config)
tf.enable_v2_behavior()



def optimize_binary_mera(chis,
                         numiters,
                         noises,
                         opt_all_layers,
                         embeddings,
                         dtype,
                         nsteps_ss,
                         use_gpu=False):
    fname = 'binary_mera_optimization'
    rootdir = os.getcwd()
    if not os.path.exists(fname):
        os.mkdir(fname)
    os.chdir(fname)

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

    with tf.device(device):
        wC, uC, _, _ = bm.run_binary_mera_optimization_TFI(
            chis=chis,
            niters=numiters,
            embeddings=embeddings,
            dtype=dtype,
            verbose=1,
            nsteps_steady_state=nsteps_ss,
            numpy_update=True,
            opt_u_after=30,
            noises=noises,
            opt_all_layers=opt_all_layers,
            filename=None)
    os.chdir(rootdir)
    return wC, uC


def load_and_optimize_binary_mera(loadname,
                                  filename,
                                  chis,
                                  numiters,
                                  noises,
                                  opt_all_layers,
                                  embeddings,
                                  nsteps_ss,
                                  use_gpu=False):

    with open(loadname, 'rb') as f:
        wC, uC = pickle.load(f)

    fname = 'binary_mera_optimization'
    rootdir = os.getcwd()
    if not os.path.exists(fname):
        os.mkdir(fname)
    os.chdir(fname)

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

    dtype = wC[-1].dtype
    num_trans_layers = len(chis)
    filename = str(datetime.date.today()) + filename \
               + 'resumed_bin_mera_opt_Nthreads{0}_chimax{1}_numtrans{2}_nss{3}'.format(
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
            uC=uC,
            filename=filename)
    os.chdir(rootdir)


def get_scaling_dims(loadname, savename, use_gpu=False, k=11):

    with open(loadname, 'rb') as f:
        wC, uC = pickle.load(f)

    fname = 'binary_mera_optimization'
    rootdir = os.getcwd()
    if not os.path.exists(fname):
        os.mkdir(fname)
    os.chdir(fname)
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

    filename = savename
    scaling_dims = {}
    # with open(filename, 'rb') as f:
    #     scaling_dims = pickle.load(f)

    with tf.device(device):
        for n in reversed(range(len(wC) - 2, len(wC))):
            print(np.array(wC[n].shape))
            if not misc_mera.all_same_chi(wC[n]):
                continue
            scaling_dims[n] = bml.get_scaling_dimensions(wC[n], uC[n], k=k)
            print(scaling_dims[n])
            with open(filename, 'wb') as f:
                pickle.dump(scaling_dims, f)


if __name__ == "__main__":
    chis = [4, 5, 6]
    numiters = [1000, 100, 400]
    noises = [1E-6, 1E-6, 1E-6]
    opt_all_layers = [True, True, True]
    embeddings = ['a', 'a', 'a']
    dtype = tf.float64
    nsteps_ss = 10
    num_scaling_dims = 11
    wC, uC = optimize_binary_mera(
        chis=chis,
        numiters=numiters,
        noises=noises,
        opt_all_layers=opt_all_layers,
        embeddings=embeddings,
        dtype=dtype,
        nsteps_ss=nsteps_ss,
        use_gpu=False)
    scaling_dims = bml.get_scaling_dimensions(
        wC[-1], uC[-1], k=num_scaling_dims)
    print()
    print(
        'first {0} eigen values of the ascending super-operator at bond dimension {1}:'
        .format(num_scaling_dims, chis[-1]))
    print(scaling_dims)
