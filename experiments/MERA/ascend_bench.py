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
import tensornetwork as tn
import experiments.MERA.binary_mera_lib as bml
import experiments.MERA.misc_mera as misc_mera
from sys import stdout
config = tf.ConfigProto()
config.intra_op_parallelism_threads = NUM_THREADS
config.inter_op_parallelism_threads = 1
tf.enable_eager_execution(config)
tf.enable_v2_behavior()

@tf.contrib.eager.defun
def left_ascending_super_operator(hamiltonian, isometry, unitary):
    """
    binary mera left ascending super operator
    Args:
        hamiltonian (tf.Tensor): hamiltonian
        isometry (tf.Tensor): isometry of the binary mera 
        unitary  (tf.Tensor): disentanlger of the mera
    Returns:
        tf.Tensor
    """

    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry)
    iso_c = net.add_node(isometry)
    iso_r = net.add_node(isometry)

    iso_l_con = net.add_node(tf.conj(isometry))
    iso_c_con = net.add_node(tf.conj(isometry))
    iso_r_con = net.add_node(tf.conj(isometry))

    op = net.add_node(hamiltonian)

    un_l = net.add_node(unitary)
    un_l_con = net.add_node(tf.conj(unitary))

    un_r = net.add_node(unitary)
    un_r_con = net.add_node(tf.conj(unitary))

    out_order = [
        iso_l_con[2], iso_c_con[2], iso_r_con[2], iso_l[2], iso_c[2], iso_r[2]
    ]
    edges = {}

    edges[1] = net.connect(iso_l[0], iso_l_con[0])
    edges[2] = net.connect(iso_r[1], iso_r_con[1])
    edges[3] = net.connect(un_l[0], op[3])
    edges[4] = net.connect(un_l[1], op[4])
    edges[5] = net.connect(un_l_con[0], op[0])
    edges[6] = net.connect(un_l_con[1], op[1])
    edges[7] = net.connect(iso_c_con[1], un_r_con[2])
    edges[8] = net.connect(iso_c_con[0], un_l_con[3])
    edges[9] = net.connect(un_r_con[0], op[2])
    edges[10] = net.connect(iso_c[1], un_r[2])
    edges[11] = net.connect(un_l[3], iso_c[0])
    edges[12] = net.connect(un_r[0], op[5])
    edges[13] = net.connect(un_r[1], un_r_con[1])
    edges[14] = net.connect(un_r[3], iso_r[0])
    edges[15] = net.connect(un_r_con[3], iso_r_con[0])
    edges[16] = net.connect(iso_l[1], un_l[2])
    edges[17] = net.connect(iso_l_con[1], un_l_con[2])

    op = net.contract_between(op, un_l)
    op = net.contract_between(op, un_l_con)

    lower = net.contract(edges[7])
    op = net.contract_between(lower, op)
    del lower

    upper = net.contract(edges[10])
    op = net.contract_between(upper, op)
    del upper

    right = net.contract(edges[2])
    op = net.contract_between(right, op)
    del right

    left = net.contract(edges[1])
    op = net.contract_between(left, op)
    del left

    op.reorder_edges(out_order)
    return op.get_tensor()


@tf.contrib.eager.defun
def right_ascending_super_operator(hamiltonian, isometry, unitary):
    """
    binary mera right ascending super operator
    Args:
        hamiltonian (tf.Tensor): hamiltonian
        isometry (tf.Tensor): isometry of the binary mera 
        unitary  (tf.Tensor): disentanlger of the mera
    Returns:
        tf.Tensor
    """

    net = tn.TensorNetwork()

    iso_l = net.add_node(isometry)
    iso_c = net.add_node(isometry)
    iso_r = net.add_node(isometry)

    iso_l_con = net.add_node(tf.conj(isometry))
    iso_c_con = net.add_node(tf.conj(isometry))
    iso_r_con = net.add_node(tf.conj(isometry))

    op = net.add_node(hamiltonian)

    un_l = net.add_node(unitary)
    un_l_con = net.add_node(tf.conj(unitary))

    un_r = net.add_node(unitary)
    un_r_con = net.add_node(tf.conj(unitary))

    out_order = [
        iso_l_con[2], iso_c_con[2], iso_r_con[2], iso_l[2], iso_c[2], iso_r[2]
    ]

    edges = {}

    edges[1] = net.connect(iso_l[0], iso_l_con[0])
    edges[2] = net.connect(iso_r[1], iso_r_con[1])
    edges[3] = net.connect(un_r[0], op[4])
    edges[4] = net.connect(un_r[1], op[5])
    edges[5] = net.connect(un_r_con[0], op[1])
    edges[6] = net.connect(un_r_con[1], op[2])
    edges[7] = net.connect(iso_c_con[0], un_l_con[3])
    edges[8] = net.connect(iso_c_con[1], un_r_con[2])
    edges[9] = net.connect(un_l_con[1], op[0])
    edges[10] = net.connect(iso_c[0], un_l[3])
    edges[11] = net.connect(un_r[2], iso_c[1])
    edges[12] = net.connect(un_l[1], op[3])
    edges[13] = net.connect(un_l[0], un_l_con[0])
    edges[14] = net.connect(un_l[2], iso_l[1])
    edges[15] = net.connect(un_l_con[2], iso_l_con[1])
    edges[16] = net.connect(iso_r[0], un_r[3])
    edges[17] = net.connect(iso_r_con[0], un_r_con[3])

    op = net.contract_between(op, un_r)
    op = net.contract_between(op, un_r_con)

    lower = net.contract(edges[7])
    op = net.contract_between(lower, op)
    del lower

    upper = net.contract(edges[10])
    op = net.contract_between(upper, op)
    del upper

    right = net.contract(edges[2])
    op = net.contract_between(right, op)
    del right

    left = net.contract(edges[1])
    op = net.contract_between(left, op)
    del left

    op.reorder_edges(out_order)
    return op.get_tensor()


@tf.contrib.eager.defun
def ascending_super_operator(ham, isometry, unitary):
    """
    binary mera ascending super operator
    Args:
        ham (tf.Tensor): hamiltonian
        isometry (tf.Tensor): isometry of the binary mera 
        unitary  (tf.Tensor): disentanlger of the mera
    Returns:
        tf.Tensor
    """
    return left_ascending_super_operator(
        ham, isometry, unitary) + right_ascending_super_operator(
            ham, isometry, unitary)

def benchmark_ascending_operator(ham, w, u, num_layers):
    """
    run benchmark for ascending super operator
    Args: 
        rhoab (tf.Tensor):  reduced densit matrix on a-b lattice
        rhoba (tf.Tensor):  reduced densit matrix on b-a lattice
        w   (tf.Tensor):  isometry
        v   (tf.Tensor):  isometry
        u   (tf.Tensor):  disentangler
        num_layers(int):  number of layers over which to descend the hamiltonian
    Returns:
        runtime (float):  the runtime
    """
    walltimes = []
    for t in range(num_layers):
        t1 = time.time()        
        ham = bml.ascending_super_operator(ham, w, u)
        walltimes.append(time.time() - t1)
    return walltimes



def run_ascending_operator_benchmark(chis=[4, 6, 8, 10, 12],
                                     num_layers=1,
                                     dtype=tf.float64,
                                     device=None):
    """
    run ascending operators benchmarks and save benchmark data in `filename`
    Args:
        chis (list):  list of bond dimensions for which to run the benchmark
        num_layers (int): number of layers over which to ascend an operator 
        dtype (tensorflow dtype): dtype to be used for the benchmark
        device (str):             device on  which the benchmark should be run
    Returns: 
       dict:  dictionary containing the walltimes
              key 'warmup' contains warmup (i.e. first run) runtimes
              key 'profile' contains subsequent runtimes
    """

    walltimes = {}
    for chi in chis:
        print('running ascending-operator benchmark for chi = {0} benchmark'.
              format(chi))
        with tf.device(device):
            w = tf.random_uniform(shape=[chi, chi, chi], dtype=dtype)
            u = tf.random_uniform(shape=[chi, chi, chi, chi], dtype=dtype)
            ham = tf.random_uniform(
                shape=[chi, chi, chi, chi, chi, chi], dtype=dtype)
            walltimes[chi] = benchmark_ascending_operator(
                ham, w, u, num_layers)
            print('wallties for chi {0}:  {1}'.format(chi, walltimes[chi]))

    return walltimes

if __name__ == "__main__":
    """
    run benchmarks for a scale-invariant binary MERA optimization
    benchmark results are stored in disc
    """
    if not tf.executing_eagerly():
        pass

    else:
        benchmarks = {
            'ascend': {
                'chis': [6],
                'dtype': tf.float64,
                'num_layers': 100
            },
        }
        use_gpu = False  #use True when running on GPU
        #list available devices
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

        if 'ascend' in benchmarks:
            filename = name + 'binary_mera_ascending_benchmark'
            for key, val in benchmarks['ascend'].items():
                if hasattr(val, 'name'):
                    val = val.name
                filename = filename + '_' + str(key) + str(val)
            filename = filename.replace(' ', '')

            fname = 'ascending_benchmarks'
            if not os.path.exists(fname):
                os.mkdir(fname)
            os.chdir(fname)
            run_ascending_operator_benchmark(
                device=specified_device_type, **benchmarks['ascend'])

            
