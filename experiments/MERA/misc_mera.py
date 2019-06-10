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
import tensornetwork as tn

@tf.contrib.eager.defun
def trace(rho):
    """
    compute the trace of `rho`
    """
    dim = len(rho.get_shape()) // 2
    net = tn.TensorNetwork()
    r = net.add_node(rho)

    edges = [net.connect(r[n], r[n + dim]) for n in range(dim)]
    out = net.contract_parallel(edges[0])
    return out.get_tensor()

@tf.contrib.eager.defun
def symmetrize(rho):
    """
    impose reflection symmetry on `rho`
    Args:
        rho (tf.Tensor)
    Returns:
        tf.Tensor:  the symmetrized version of `rho`
    """
    dim = len(rho.get_shape()) // 2
    inds_1 = [n for n in range(dim)]
    inds_2 = [n + dim for n in range(dim)]
    indices = inds_2 + inds_1
    return 1 / 2 * (rho + tf.conj(tf.transpose(rho, indices)))


@tf.contrib.eager.defun
def scalar_product(bottom, top):
    """
    calculate the Hilbert-schmidt inner product between `bottom` and `top'
    Args:
        bottom (tf.Tensor)
        top (tf.Tensor)
    Returns:
        tf.Tensor:  the inner product
    """

    net = tn.TensorNetwork()
    b = net.add_node(tf.conj(bottom))
    t = net.add_node(top)
    edges = [net.connect(b[n], t[n]) for n in range(len(bottom.get_shape()))]
    out = net.contract_between(b, t)
    return out.get_tensor()

def pad_tensor(tensor, new_shape):
    """
    pad `tensor` with zeros to shape `new_shape`
    """
    paddings = np.zeros((len(tensor.get_shape()), 2)).astype(np.int32)
    for n in range(len(new_shape)):
        paddings[n, 1] = max(new_shape[n] - tensor.get_shape()[n], 0)
    return tf.pad(tensor, paddings)


def all_same_chi(*tensors):
    """
    test if all input arguments have the same bond dimension
    """
    chis = [t.get_shape()[n] for t in tensors for n in range(len(t.get_shape()))]
    return np.all([c == chis[0] for c in chis])

@tf.contrib.eager.defun
def u_update_svd(wIn):
    """
    obtain the update to the disentangler using tf.svd
    """
    shape = tf.shape(wIn)
    st, ut, vt = tf.linalg.svd(
        tf.reshape(wIn, (shape[0] * shape[1], shape[2] * shape[3])),
        full_matrices=False)
    return -tf.reshape(tn.ncon([ut, tf.conj(vt)], [[-1, 1], [-2, 1]]), shape)

#@tf.contrib.eager.defun
def u_update_svd_numpy(wIn):
    """
    obtain the update to the disentangler using numpy svd
    """
    shape = tf.shape(wIn)
    ut, st, vt = np.linalg.svd(
        tf.reshape(wIn, (shape[0] * shape[1], shape[2] * shape[3])),
        full_matrices=False)
    return -tf.reshape(tn.ncon([ut, vt], [[-1, 1], [1, -2]]), shape)

@tf.contrib.eager.defun
def w_update_svd(wIn):
    """
    obtain the update to the isometry using tf.tensor
    """
    shape = wIn.get_shape()
    st, ut, vt = tf.linalg.svd(
        tf.reshape(wIn, (shape[0] * shape[1], shape[2])), full_matrices=False)
    return -tf.reshape(tn.ncon([ut, tf.conj(vt)], [[-1, 1], [-2, 1]]), shape)

def w_update_svd_numpy(wIn):
    """
    obtain the update to the isometry using numpy svd
    """
    shape = wIn.get_shape()
    ut, st, vt = np.linalg.svd(
        tf.reshape(wIn, (shape[0] * shape[1], shape[2])), full_matrices=False)
    return -tf.reshape(tn.ncon([ut, vt], [[-1, 1], [1, -2]]), shape)

def skip_layer(isometry):
    if isometry.get_shape()[2] >= (isometry.get_shape()[0] * isometry.get_shape()[1]):
        return True
    else:
        return False

