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
import tensornetwork
from tensornetwork import TensorNetwork
from tensornetwork.sparse import lanczos
import numpy as np
import tensorflow as tf
import torch
import pytest
from jax.config import config
from tensornetwork.backends import backend_factory

config.update("jax_enable_x64", True)
tf.compat.v1.enable_v2_behavior()

np_dtypes = [np.float64, np.complex128]
tf_dtypes = [tf.float64, tf.complex128]
torch_dtypes = [torch.float64]
jax_dtypes = [np.float64, np.complex128]


@pytest.mark.parametrize("backend,dtype", [
    *list(zip(['numpy'] * len(np_dtypes), np_dtypes)),
    *list(zip(['tensorflow'] * len(tf_dtypes), tf_dtypes)),
    *list(zip(['pytorch'] * len(torch_dtypes), torch_dtypes)),
    *list(zip(['jax'] * len(jax_dtypes), jax_dtypes)),
])
def test_eigsh_lanczos_1(backend, dtype):
  be = backend_factory.get_backend(backend, dtype)
  D = 16
  np.random.seed(10)
  tmp = be.randn((D, D))
  Hmat = tmp + be.transpose(be.conj(tmp), (1, 0))
  H = be.reshape(Hmat, (4, 4, 4, 4))
  Hmat = np.array(Hmat)

  def mv(x, backend):
    net = TensorNetwork(backend=backend)
    n1 = net.add_node(H)
    n2 = net.add_node(x)
    n1[2] ^ n2[0]
    n1[3] ^ n2[1]
    out_order = [n1[0], n1[1]]
    result = n1 @ n2
    result.reorder_edges(out_order)
    return result.tensor

  def vv(a, b, backend):
    net = TensorNetwork(backend=backend)
    n1 = net.add_node(net.backend.conj(a))
    n2 = net.add_node(b)
    n1[0] ^ n2[0]
    n1[1] ^ n2[1]
    result = n1 @ n2
    return result.tensor

  A = lanczos.LinearOperator(mv, ((4, 4), (4, 4)), dtype=dtype, backend=backend)
  vv = lanczos.ScalarProduct(vv, dtype=dtype, backend=backend)
  eta1, U1 = lanczos.eigsh_lanczos(A, vv)
  eta2, U2 = np.linalg.eigh(Hmat)
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("backend,dtype", [
    *list(zip(['numpy'] * len(np_dtypes), np_dtypes)),
    *list(zip(['tensorflow'] * len(tf_dtypes), tf_dtypes)),
    *list(zip(['pytorch'] * len(torch_dtypes), torch_dtypes)),
    *list(zip(['jax'] * len(jax_dtypes), jax_dtypes)),
])
def test_eigsh_lanczos_2(backend, dtype):
  be = backend_factory.get_backend(backend, dtype)
  D = 16
  np.random.seed(10)
  tmp = be.randn((D, D))
  Hmat = tmp + be.transpose(be.conj(tmp), (1, 0))
  H = be.reshape(Hmat, (4, 4, 4, 4))
  Hmat = np.array(Hmat)

  def mv(x):
    net = TensorNetwork(backend=backend)
    n1 = net.add_node(H)
    n2 = net.add_node(x)
    n1[2] ^ n2[0]
    n1[3] ^ n2[1]
    out_order = [n1[0], n1[1]]
    result = n1 @ n2
    result.reorder_edges(out_order)
    return result.tensor

  def vv(a, b):
    net = TensorNetwork(backend=backend)
    n1 = net.add_node(net.backend.conj(a))
    n2 = net.add_node(b)
    n1[0] ^ n2[0]
    n1[1] ^ n2[1]
    result = n1 @ n2
    return result.tensor

  A = lanczos.LinearOperator(mv, ((4, 4), (4, 4)), dtype=dtype, backend=backend)
  vv = lanczos.ScalarProduct(vv, dtype=dtype, backend=backend)
  eta1, U1 = lanczos.eigsh_lanczos(A, vv)
  eta2, U2 = np.linalg.eigh(Hmat)
  v2 = U2[:, 0]
  v2 = v2 / sum(v2)
  v1 = np.reshape(U1[0], (D))
  v1 = v1 / sum(v1)
  np.testing.assert_allclose(eta1[0], min(eta2))
  np.testing.assert_allclose(v1, v2)


@pytest.mark.parametrize("backend,dtype", [
    *list(zip(['numpy'] * len(np_dtypes), np_dtypes)),
    *list(zip(['tensorflow'] * len(tf_dtypes), tf_dtypes)),
    *list(zip(['pytorch'] * len(torch_dtypes), torch_dtypes)),
    *list(zip(['jax'] * len(jax_dtypes), jax_dtypes)),
])
def test_LinearOperator_1(backend, dtype):
  # pylint: disable=unused-argument
  # pylint: disable=unused-argument
  def mv(x, backend, c):
    pass

  with pytest.raises(ValueError):
    lanczos.LinearOperator(mv, ((4, 4), (4, 4)), dtype=dtype, backend=backend)


@pytest.mark.parametrize("backend,dtype", [
    *list(zip(['numpy'] * len(np_dtypes), np_dtypes)),
    *list(zip(['tensorflow'] * len(tf_dtypes), tf_dtypes)),
    *list(zip(['pytorch'] * len(torch_dtypes), torch_dtypes)),
    *list(zip(['jax'] * len(jax_dtypes), jax_dtypes)),
])
def test_LinearOperator_2(backend, dtype):
  # pylint: disable=unused-argument
  # pylint: disable=unused-argument
  def mv(backend, x):
    pass

  with pytest.raises(ValueError):
    lanczos.LinearOperator(mv, ((4, 4), (4, 4)), dtype=dtype, backend=backend)
