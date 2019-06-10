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
def left_ascending_super_operator(ham, isometry, unitary):
    inds_right_ul = [1, 2, 13, 5]
    inds_right_ur = [6, 9, 7, 16]
    inds_right_ul_c = [3, 4, 14, 10]
    inds_right_ur_c = [8, 9, 11, 17]

    inds_right_iso_l = [12, 13, -4]
    inds_right_iso_c = [5, 7, -5]
    inds_right_iso_r = [16, 15, -6]
    inds_right_iso_l_c = [12, 14, -1]
    inds_right_iso_c_c = [10, 11, -2]
    inds_right_iso_r_c = [17, 15, -3]

    inds_right_ham = [3, 4, 8, 1, 2, 6]

    hright = tn.ncon([
        isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham
    ], [
        inds_right_iso_l, inds_right_iso_c, inds_right_iso_r,
        inds_right_iso_l_c, inds_right_iso_c_c, inds_right_iso_r_c,
        inds_right_ul, inds_right_ur, inds_right_ul_c, inds_right_ur_c,
        inds_right_ham
    ])
    return hright


@tf.contrib.eager.defun
def right_ascending_super_operator(ham, isometry, unitary):
    inds_left_ul = [8, 6, 13, 7]
    inds_left_ur = [1, 2, 5, 16]
    inds_left_ul_c = [8, 9, 14, 10]
    inds_left_ur_c = [3, 4, 11, 17]
    inds_left_iso_l = [12, 13, -4]
    inds_left_iso_c = [7, 5, -5]
    inds_left_iso_r = [16, 15, -6]
    inds_left_iso_l_c = [12, 14, -1]
    inds_left_iso_c_c = [10, 11, -2]
    inds_left_iso_r_c = [17, 15, -3]
    inds_left_ham = [9, 3, 4, 6, 1, 2]

    hleft = tn.ncon([
        isometry, isometry, isometry,
        tf.conj(isometry),
        tf.conj(isometry),
        tf.conj(isometry), unitary, unitary,
        tf.conj(unitary),
        tf.conj(unitary), ham
    ], [
        inds_left_iso_l, inds_left_iso_c, inds_left_iso_r, inds_left_iso_l_c,
        inds_left_iso_c_c, inds_left_iso_r_c, inds_left_ul, inds_left_ur,
        inds_left_ul_c, inds_left_ur_c, inds_left_ham
    ])
    return hleft


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
    chi = 6
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

