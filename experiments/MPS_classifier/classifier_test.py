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

import numpy as np
import tensorflow as tf
from experiments.MPS_classifier import classifier
tf.enable_v2_behavior()


def test_random_initializer_shapes():
  x = classifier.random_initializer(2, 3)
  assert x.shape == (2, 3, 3)
  y = classifier.random_initializer(2, 3, boundary=True)
  assert y.shape == (2, 3)

def test_environment_shapes():
  env = classifier.Environment(5, 2, 7)
  assert list(env.vector.shape) == [2, 7]
  assert list(env.matrices.shape) == [4, 2, 7, 7]

def test_environment_network():
  env = classifier.Environment(5, 2, 7)
  data = tf.cast(np.random.random([20, 5, 2]), dtype=tf.float32)
  net, var_nodes, data_nodes = env.create_network(data[:, 1:], data[:, 0])
  net.check_correct()
  np.testing.assert_allclose(data_nodes[0].tensor.numpy(), data[:, 0])
  np.testing.assert_allclose(data_nodes[1].tensor.numpy(), data[:, 1:])

def test_environment_predict():
  env = classifier.Environment(5, 2, 7)
  data = tf.cast(np.random.random([20, 5, 2]), dtype=tf.float32)
  # Compare with serialized contraction
  pred = tf.einsum("bs,sr->br", data[:, 0], env.vector)
  for i in range(4):
    pred = tf.einsum("bl,bs,slr->br", pred, data[:, i + 1], env.matrices[i])
  np.testing.assert_allclose(env.predict(data[:, 1:], data[:, 0]).numpy(),
                             pred.numpy(), atol=1e-5)

def test_mps_shapes():
  mps = classifier.MatrixProductState(11, 10, 2, 7)
  assert list(mps.tensors[0].shape) == [2, 7]
  assert list(mps.tensors[1].shape) == [4, 2,  7, 7]
  assert list(mps.tensors[2].shape) == [10, 7, 7]
  assert list(mps.tensors[3].shape) == [4, 2, 7, 7]
  assert list(mps.tensors[4].shape) == [2, 7]

def test_calculate_flx():
  mps = classifier.MatrixProductState(11, 10, 2, 7)
  data = tf.cast(np.random.random([20, 10, 2]), dtype=mps.dtype)
  # Compare with serialized contraction
  flx_left = tf.einsum("bs,sr->br", data[:, 0], mps.tensors[0])
  flx_right = tf.einsum("bs,sr->br", data[:, -1], mps.tensors[4])
  for i in range(4):
    flx_left = tf.einsum("bl,bs,slr->br", flx_left, data[:, i + 1],
                         mps.tensors[1][i])
    flx_right = tf.einsum("bl, bs,slr->br", flx_right, data[:, 8 - i],
                          mps.tensors[3][i])
  flx = tf.einsum("bl,olr,br->bo", flx_left, mps.tensors[2], flx_right)
  np.testing.assert_allclose(flx.numpy(), mps.flx(data).numpy(), atol=1e-5)
