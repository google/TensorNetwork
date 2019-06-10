from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import tensornetwork as tn
import numpy as np
from sys import stdout
tf.enable_eager_execution()
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

if __name__ == "__main__":
    """
    run benchmarks for a scale-invariant binary MERA optimization
    benchmark results are stored in disc
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
        name = 'GPU'
    else:
        specified_device_type = CPU
        name = 'CPU'
    chi = 16
    dtype = tf.float64
    with tf.device(specified_device_type):        
        w = tf.random_uniform(shape=[chi, chi, chi], dtype=dtype)
        u = tf.random_uniform(shape=[chi, chi, chi, chi], dtype=dtype)
        ham = tf.random_uniform(
            shape=[chi, chi, chi, chi, chi, chi], dtype=dtype)

        for t in range(100):
            t1 = time.time()        
            ascending_super_operator(ham, w, u)
            print(time.time() - t1)

