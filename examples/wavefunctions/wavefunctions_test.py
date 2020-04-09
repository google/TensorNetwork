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

import pytest
import numpy as np
import tensorflow as tf
import tensornetwork as tn
from examples.wavefunctions import wavefunctions


@pytest.mark.parametrize("num_sites", [2, 3, 4])
def test_expval(num_sites):
  op = np.kron(np.array([[1.0, 0.0], [0.0, -1.0]]), np.eye(2)).reshape([2] * 4)
  op = tf.convert_to_tensor(op)
  for j in range(num_sites):
    psi = np.zeros([2] * num_sites)
    psi_vec = psi.reshape((2**num_sites,))
    psi_vec[2**j] = 1.0
    psi = tf.convert_to_tensor(psi)
    for i in range(num_sites):
      res = wavefunctions.expval(psi, op, i, pbc=True)
      if i == num_sites - 1 - j:
        np.testing.assert_allclose(res, -1.0)
      else:
        np.testing.assert_allclose(res, 1.0)


@pytest.mark.parametrize("num_sites", [2, 3, 4])
def test_apply_op(num_sites):
  psi1 = np.zeros([2] * num_sites)
  psi1_vec = psi1.reshape((2**num_sites,))
  psi1_vec[0] = 1.0
  psi1 = tf.convert_to_tensor(psi1)

  for j in range(num_sites):
    psi2 = np.zeros([2] * num_sites)
    psi2_vec = psi2.reshape((2**num_sites,))
    psi2_vec[2**j] = 1.0
    psi2 = tf.convert_to_tensor(psi2)

    opX = tf.convert_to_tensor(np.array([[0.0, 1.0], [1.0, 0.0]]))
    psi2 = wavefunctions.apply_op(psi2, opX, num_sites - 1 - j)

    res = wavefunctions.inner(psi1, psi2)
    np.testing.assert_allclose(res, 1.0)


@pytest.mark.parametrize("num_sites,phys_dim,graph",
                         [(2, 3, False), (2, 3, True), (5, 2, False)])
def test_evolve_trotter(num_sites, phys_dim, graph):
  tf.random.set_seed(10)
  psi = tf.complex(
      tf.random.normal([phys_dim] * num_sites, dtype=tf.float64),
      tf.random.normal([phys_dim] * num_sites, dtype=tf.float64))
  h = tf.complex(
      tf.random.normal((phys_dim**2, phys_dim**2), dtype=tf.float64),
      tf.random.normal((phys_dim**2, phys_dim**2), dtype=tf.float64))
  h = 0.5 * (h + tf.linalg.adjoint(h))
  h = tf.reshape(h, (phys_dim, phys_dim, phys_dim, phys_dim))
  H = [h] * (num_sites - 1)

  norm1 = wavefunctions.inner(psi, psi)
  en1 = sum(wavefunctions.expval(psi, H[i], i) for i in range(num_sites - 1))

  if graph:
    psi, t = wavefunctions.evolve_trotter_defun(psi, H, 0.001, 10)
  else:
    psi, t = wavefunctions.evolve_trotter(psi, H, 0.001, 10)

  norm2 = wavefunctions.inner(psi, psi)
  en2 = sum(wavefunctions.expval(psi, H[i], i) for i in range(num_sites - 1))

  np.testing.assert_allclose(t, 0.01)
  np.testing.assert_almost_equal(norm1 / norm2, 1.0)
  np.testing.assert_almost_equal(en1 / en2, 1.0, decimal=2)


@pytest.mark.parametrize("num_sites,phys_dim,graph",
                         [(2, 3, False), (2, 3, True), (5, 2, False)])
def test_evolve_trotter_euclidean(num_sites, phys_dim, graph):
  tf.random.set_seed(10)
  psi = tf.complex(
      tf.random.normal([phys_dim] * num_sites, dtype=tf.float64),
      tf.random.normal([phys_dim] * num_sites, dtype=tf.float64))
  h = tf.complex(
      tf.random.normal((phys_dim**2, phys_dim**2), dtype=tf.float64),
      tf.random.normal((phys_dim**2, phys_dim**2), dtype=tf.float64))
  h = 0.5 * (h + tf.linalg.adjoint(h))
  h = tf.reshape(h, (phys_dim, phys_dim, phys_dim, phys_dim))
  H = [h] * (num_sites - 1)

  norm1 = wavefunctions.inner(psi, psi)
  en1 = sum(wavefunctions.expval(psi, H[i], i) for i in range(num_sites - 1))

  if graph:
    psi, t = wavefunctions.evolve_trotter_defun(psi, H, 0.1, 10, euclidean=True)
  else:
    psi, t = wavefunctions.evolve_trotter(psi, H, 0.1, 10, euclidean=True)

  norm2 = wavefunctions.inner(psi, psi)
  en2 = sum(wavefunctions.expval(psi, H[i], i) for i in range(num_sites - 1))

  np.testing.assert_allclose(t, 1.0)
  np.testing.assert_almost_equal(norm2, 1.0)
  assert en2.numpy() / norm2.numpy() < en1.numpy() / norm1.numpy()
