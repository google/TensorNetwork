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
"""
miscellaneous functions needed for MERA optimization
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensornetwork.ncon_interface as ncon

@tf.contrib.eager.defun
def trace(rho):
    dim = len(rho.shape) // 2
    inds = [n + 1 for n in range(dim)]
    inds = list(range(dim))
    return ncon.ncon([rho], [inds + inds])


@tf.contrib.eager.defun
def symmetrize(rho):
    dim = len(rho.shape) // 2
    inds_1 = [n for n in range(dim)]
    inds_2 = [n + dim for n in range(dim)]
    indices = inds_2 + inds_1
    return 1 / 2 * (rho + tf.conj(tf.transpose(rho, indices)))


@tf.contrib.eager.defun
def scalar_product(bottom, top):
    inds = list(range(len(top.shape)))
    return ncon.ncon([tf.conj(bottom), top], [inds, inds])


def pad_tensor(tensor, new_shape):
    paddings = np.zeros((len(tensor.shape), 2)).astype(np.int32)
    for n in range(len(new_shape)):
        paddings[n, 1] = max(new_shape[n] - tensor.shape[n], 0)
    return tf.pad(tensor, paddings)


def all_same_chi(*tensors):
    chis = [t.shape[n] for t in tensors for n in range(len(t.shape))]
    return np.all([c == chis[0] for c in chis])


@tf.contrib.eager.defun
def u_update_svd(wIn):
    shape = wIn.shape
    st, ut, vt = tf.linalg.svd(
        tf.reshape(wIn, (shape[0] * shape[1], shape[2] * shape[3])),
        full_matrices=False)
    return -tf.reshape(ncon.ncon([ut, tf.conj(vt)], [[-1, 1], [-2, 1]]), shape)


def u_update_svd_numpy(wIn):
    shape = wIn.shape
    ut, st, vt = np.linalg.svd(
        tf.reshape(wIn, (shape[0] * shape[1], shape[2] * shape[3])),
        full_matrices=False)
    return -tf.reshape(ncon.ncon([ut, vt], [[-1, 1], [1, -2]]), shape)


@tf.contrib.eager.defun
def w_update_svd(wIn):
    shape = wIn.shape
    st, ut, vt = tf.linalg.svd(
        tf.reshape(wIn, (shape[0] * shape[1], shape[2])), full_matrices=False)
    return -tf.reshape(ncon.ncon([ut, tf.conj(vt)], [[-1, 1], [-2, 1]]), shape)


def w_update_svd_numpy(wIn):
    shape = wIn.shape
    ut, st, vt = np.linalg.svd(
        tf.reshape(wIn, (shape[0] * shape[1], shape[2])), full_matrices=False)
    return -tf.reshape(ncon.ncon([ut, vt], [[-1, 1], [1, -2]]), shape)


def skip_layer(isometry):
    if isometry.shape[2] >= (isometry.shape[0] * isometry.shape[1]):
        return True
    else:
        return False
