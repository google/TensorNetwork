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
import pytest
import numpy as np
import tensornetwork as tn
from tensornetwork.backends import backend_factory
from tensornetwork.matrixproductstates.base_mps import BaseMPS
import tensorflow as tf

from jax.config import config

config.update("jax_enable_x64", True)
tf.compat.v1.enable_v2_behavior()


@pytest.fixture(
    name="backend_dtype_values",
    params=[('numpy', np.float64), ('numpy', np.complex128),
            ('tensorflow', np.float64), ('tensorflow', np.complex128),
            ('pytorch', np.float64), ('jax', np.float64)])
def backend_dtype(request):
  return request.param


def get_random_np(shape, dtype, seed=0):
  np.random.seed(seed)  #get the same tensors every time you call this function
  if dtype is np.complex64:
    return np.random.randn(*shape).astype(
        np.float32) + 1j * np.random.randn(*shape).astype(np.float32)
  if dtype is np.complex128:
    return np.random.randn(*shape).astype(
        np.float64) + 1j * np.random.randn(*shape).astype(np.float64)
  return np.random.randn(*shape).astype(dtype)


def test_normalization(backend):
  D, d, N = 10, 2, 10
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  mps = BaseMPS(tensors, center_position=0, backend=backend)
  mps.position(len(mps) - 1)
  Z = mps.position(0, normalize=True)
  np.testing.assert_allclose(Z, 1.0)


def test_backend_initialization(backend):
  be = backend_factory.get_backend(backend)
  D, d, N = 10, 2, 10
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  mps = BaseMPS(tensors, center_position=0, backend=be)
  mps.position(len(mps) - 1)
  Z = mps.position(0, normalize=True)
  np.testing.assert_allclose(Z, 1.0)


def test_backend_initialization_raises(backend):
  be = backend_factory.get_backend(backend)
  D, d, N = 10, 2, 10
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  with pytest.raises(
      ValueError,
      match="`center_position = 10` is different from `None` and "
      "not between 0 <= center_position < 10"):
    BaseMPS(tensors, center_position=N, backend=be)
  with pytest.raises(
      ValueError,
      match="`center_position = -1` is different from `None` and "
      "not between 0 <= center_position < 10"):
    BaseMPS(tensors, center_position=-1, backend=be)


def test_left_orthonormalization(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = BaseMPS(tensors, center_position=N - 1, backend=backend)
  mps.position(0)
  mps.position(len(mps) - 1)
  assert all([
      abs(mps.check_orthonormality('left', site)) < 1E-12
      for site in range(len(mps))
  ])


def test_right_orthonormalization(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]
  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = BaseMPS(tensors, center_position=0, backend=backend)

  mps.position(len(mps) - 1)
  mps.position(0)
  assert all([
      abs(mps.check_orthonormality('right', site)) < 1E-12
      for site in range(len(mps))
  ])


def test_apply_one_site_gate(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = BaseMPS(tensors, center_position=0, backend=backend)
  tensor = mps.tensors[5]
  gate = get_random_np((2, 2), dtype)

  mps.apply_one_site_gate(gate, 5)
  actual = np.transpose(np.tensordot(tensor, gate, ([1], [1])), (0, 2, 1))
  np.testing.assert_allclose(mps.tensors[5], actual)


def test_apply_two_site_gate(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = BaseMPS(tensors, center_position=0, backend=backend)
  gate = get_random_np((2, 2, 2, 2), dtype)
  tensor1 = mps.tensors[5]
  tensor2 = mps.tensors[6]

  mps.apply_two_site_gate(gate, 5, 6)
  tmp = np.tensordot(tensor1, tensor2, ([2], [0]))
  actual = np.transpose(np.tensordot(tmp, gate, ([1, 2], [2, 3])), (0, 2, 3, 1))
  node1 = tn.Node(mps.tensors[5], backend=backend)
  node2 = tn.Node(mps.tensors[6], backend=backend)

  node1[2] ^ node2[0]
  order = [node1[0], node1[1], node2[1], node2[2]]
  res = tn.contract_between(node1, node2)
  res.reorder_edges(order)
  np.testing.assert_allclose(res.tensor, actual)


def test_position_raises_error(backend):
  D, d, N = 10, 2, 10
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  mps = BaseMPS(tensors, center_position=0, backend=backend)
  with pytest.raises(
      ValueError, match="site = -1 not between values"
      " 0 < site < N = 10"):
    mps.position(-1)
  with pytest.raises(
      ValueError, match="site = 11 not between values"
      " 0 < site < N = 10"):
    mps.position(11)
  mps = BaseMPS(tensors, center_position=None, backend=backend)
  with pytest.raises(
      ValueError,
      match="BaseMPS.center_position is"
      " `None`, cannot shift `center_position`."
      "Reset `center_position` manually or use `canonicalize`"):
    mps.position(1)


def test_position_no_normalization(backend):
  D, d, N = 4, 2, 6
  tensors = [np.ones((1, d, D))] + [np.ones((D, d, D)) for _ in range(N - 2)
                                   ] + [np.ones((D, d, 1))]
  mps = BaseMPS(tensors, center_position=0, backend=backend)
  Z = mps.position(len(mps) - 1, normalize=False)
  np.testing.assert_allclose(Z, 8192.0)


def test_position_shift_left(backend):
  D, d, N = 4, 2, 6
  tensors = [np.ones((1, d, D))] + [np.ones((D, d, D)) for _ in range(N - 2)
                                   ] + [np.ones((D, d, 1))]
  mps = BaseMPS(tensors, center_position=int(N / 2), backend=backend)
  Z = mps.position(0, normalize=True)
  np.testing.assert_allclose(Z, 2.828427)


def test_position_shift_right(backend):
  D, d, N = 4, 2, 6
  tensors = [np.ones((1, d, D))] + [np.ones((D, d, D)) for _ in range(N - 2)
                                   ] + [np.ones((D, d, 1))]
  mps = BaseMPS(tensors, center_position=int(N / 2), backend=backend)
  Z = mps.position(N - 1, normalize=True)
  np.testing.assert_allclose(Z, 2.828427)


def test_position_no_shift(backend):
  D, d, N = 4, 2, 6
  tensors = [np.ones((1, d, D))] + [np.ones((D, d, D)) for _ in range(N - 2)
                                   ] + [np.ones((D, d, 1))]
  mps = BaseMPS(tensors, center_position=int(N / 2), backend=backend)
  Z = mps.position(int(N / 2), normalize=True)
  np.testing.assert_allclose(Z, 5.656854)


def test_position_no_shift_no_normalization(backend):
  D, d, N = 4, 2, 6
  tensors = [np.ones((1, d, D))] + [np.ones((D, d, D)) for _ in range(N - 2)
                                   ] + [np.ones((D, d, 1))]
  mps = BaseMPS(tensors, center_position=int(N / 2), backend=backend)
  Z = mps.position(int(N / 2), normalize=False)
  np.testing.assert_allclose(Z, 5.656854)


def test_different_dtypes_raises_error():
  D, d = 4, 2
  tensors = [
      np.ones((1, d, D), dtype=np.float64),
      np.ones((D, d, D), dtype=np.complex64)
  ]
  with pytest.raises(TypeError):
    BaseMPS(tensors, backend='numpy')

  _tensors = [
      np.ones((1, d, D), dtype=np.float64),
      np.ones((D, d, D), dtype=np.float64)
  ]

  mps = BaseMPS(_tensors, backend='numpy')
  mps.tensors = tensors
  with pytest.raises(TypeError):
    mps.dtype


def test_not_implemented():
  D, d = 4, 2
  tensors = [np.ones((1, d, D)), np.ones((D, d, D))]
  mps = BaseMPS(tensors, backend='numpy')
  with pytest.raises(NotImplementedError):
    mps.save('tmp')
  with pytest.raises(NotImplementedError):
    mps.right_envs([0])
  with pytest.raises(NotImplementedError):
    mps.left_envs([0])
  with pytest.raises(NotImplementedError):
    mps.canonicalize()


def test_physical_dimensions(backend):
  D = 3
  tensors = [np.ones((1, 2, D)), np.ones((D, 3, D)), np.ones((D, 4, 1))]
  mps = BaseMPS(tensors, backend=backend)
  assert mps.physical_dimensions == [2, 3, 4]


def test_apply_transfer_operator_left(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)

  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mat = backend.convert_to_tensor(
      np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64))
  mps = BaseMPS(tensors, backend=backend)

  expected = np.array([[74., 58., 38.], [78., 146., 102.], [38., 114., 74.]])
  actual = mps.apply_transfer_operator(site=3, direction=1, matrix=mat)
  np.testing.assert_allclose(actual, expected)
  actual = mps.apply_transfer_operator(site=3, direction="l", matrix=mat)
  np.testing.assert_allclose(actual, expected)
  actual = mps.apply_transfer_operator(site=3, direction="left", matrix=mat)
  np.testing.assert_allclose(actual, expected)


def test_apply_transfer_operator_right(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)

  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mat = backend.convert_to_tensor(
      np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64))
  mps = BaseMPS(tensors, backend=backend)
  expected = np.array([[80., -20., 128.], [-20., 10., -60.], [144., -60.,
                                                              360.]])
  actual = mps.apply_transfer_operator(site=3, direction=-1, matrix=mat)
  np.testing.assert_allclose(actual, expected)
  actual = mps.apply_transfer_operator(site=3, direction="r", matrix=mat)
  np.testing.assert_allclose(actual, expected)
  actual = mps.apply_transfer_operator(site=3, direction="right", matrix=mat)
  np.testing.assert_allclose(actual, expected)


def test_apply_transfer_operator_invalid_direction_raises_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)

  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mat = backend.convert_to_tensor(
      np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64))
  mps = BaseMPS(tensors, backend=backend)
  with pytest.raises(ValueError):
    mps.apply_transfer_operator(site=3, direction=0, matrix=mat)
  with pytest.raises(ValueError):
    mps.apply_transfer_operator(site=3, direction="keft", matrix=mat)


def test_measure_local_operator_value_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)

  tensors = 6 * [backend.convert_to_tensor(tensor)]
  operator = backend.convert_to_tensor(
      np.array([[1, -1], [-1, 1]], dtype=np.float64))
  mps = BaseMPS(tensors, backend=backend)
  with pytest.raises(ValueError):
    mps.measure_local_operator(ops=2 * [operator], sites=[1, 2, 3])


def test_measure_two_body_correlator_value_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)

  tensors = 6 * [backend.convert_to_tensor(tensor)]
  operator = backend.convert_to_tensor(
      np.array([[1, -1], [-1, 1]], dtype=np.float64))
  mps = BaseMPS(tensors, backend=backend)
  with pytest.raises(ValueError):
    mps.measure_two_body_correlator(
        op1=operator, op2=operator, site1=-1, sites2=[2])


def test_get_tensor(backend):
  backend = backend_factory.get_backend(backend)
  tensor1 = np.ones((2, 3, 2), dtype=np.float64)
  tensor2 = 2 * np.ones((2, 3, 2), dtype=np.float64)
  tensors = [tensor1, tensor2]
  mps = BaseMPS(tensors, backend=backend)
  np.testing.assert_allclose(mps.get_tensor(0), tensor1)
  np.testing.assert_allclose(mps.get_tensor(1), tensor2)


def test_get_tensor_connector_matrix(backend):
  backend = backend_factory.get_backend(backend)
  tensor1 = np.ones((2, 3, 2), dtype=np.float64)
  tensor2 = 2 * np.ones((2, 3, 2), dtype=np.float64)
  connector = backend.convert_to_tensor(np.ones((2, 2), dtype=np.float64))
  tensors = [tensor1, tensor2]
  mps = BaseMPS(tensors, backend=backend, connector_matrix=connector)
  np.testing.assert_allclose(mps.get_tensor(0), tensor1)
  np.testing.assert_allclose(mps.get_tensor(1), 2 * tensor2)


def test_get_tensor_raises_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor1 = np.ones((2, 3, 2), dtype=np.float64)
  tensor2 = 2 * np.ones((2, 3, 2), dtype=np.float64)
  tensors = [tensor1, tensor2]
  mps = BaseMPS(tensors, backend=backend)
  with pytest.raises(ValueError):
    mps.get_tensor(site=-1)
  with pytest.raises(IndexError):
    mps.get_tensor(site=3)


def test_check_canonical(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)
  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mps = BaseMPS(tensors, backend=backend, center_position=2)
  np.testing.assert_allclose(mps.check_canonical(), 71.714713)


def test_check_normality_raises_value_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.ones((2, 3, 2), dtype=np.float64)
  tensors = [tensor]
  mps = BaseMPS(tensors, backend=backend)
  with pytest.raises(ValueError):
    mps.check_orthonormality(which="keft", site=0)


def test_apply_two_site_gate_2(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)
  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mps = BaseMPS(tensors, backend=backend, center_position=2)
  gate = backend.convert_to_tensor(
      np.array([[[[0., 1.], [0., 0.]], [[1., 0.], [0., 0.]]],
                [[[0., 0.], [0., 1.]], [[0., 0.], [1., 0.]]]],
               dtype=np.float64))
  actual = mps.apply_two_site_gate(
      gate=gate, site1=1, site2=2, max_singular_values=1)
  np.testing.assert_allclose(actual[0], 9.133530)
  expected = np.array([[5.817886], [9.039142]])
  np.testing.assert_allclose(np.abs(mps.tensors[1][0]), expected, rtol=1e-04)
  expected = np.array([[0.516264, 0.080136, 0.225841],
                       [0.225841, 0.59876, 0.516264]])
  np.testing.assert_allclose(np.abs(mps.tensors[2][0]), expected, rtol=1e-04)


def test_apply_two_site_wrong_gate_raises_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)
  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mps = BaseMPS(tensors, backend=backend, center_position=2)
  gate1 = backend.convert_to_tensor(np.ones((2, 2, 2), dtype=np.float64))
  gate2 = backend.convert_to_tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float64))
  with pytest.raises(ValueError):
    mps.apply_two_site_gate(gate=gate1, site1=1, site2=2)
  with pytest.raises(ValueError):
    mps.apply_two_site_gate(gate=gate2, site1=1, site2=2)


def test_apply_two_site_wrong_site1_raises_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)
  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mps = BaseMPS(tensors, backend=backend, center_position=2)
  gate = backend.convert_to_tensor(np.ones((2, 2, 2, 2), dtype=np.float64))
  with pytest.raises(ValueError):
    mps.apply_two_site_gate(gate=gate, site1=-1, site2=2)
  with pytest.raises(ValueError):
    mps.apply_two_site_gate(gate=gate, site1=6, site2=2)


def test_apply_two_site_wrong_site2_raises_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)
  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mps = BaseMPS(tensors, backend=backend, center_position=2)
  gate = backend.convert_to_tensor(np.ones((2, 2, 2, 2), dtype=np.float64))
  with pytest.raises(ValueError):
    mps.apply_two_site_gate(gate=gate, site1=0, site2=0)
  with pytest.raises(ValueError):
    mps.apply_two_site_gate(gate=gate, site1=0, site2=6)


def test_apply_two_site_wrong_site1_site2_raises_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)
  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mps = BaseMPS(tensors, backend=backend, center_position=2)
  gate = backend.convert_to_tensor(np.ones((2, 2, 2, 2), dtype=np.float64))
  with pytest.raises(ValueError):
    mps.apply_two_site_gate(gate=gate, site1=2, site2=2)
  with pytest.raises(ValueError):
    mps.apply_two_site_gate(gate=gate, site1=2, site2=4)


def test_apply_two_site_max_singular_value_not_center_raises_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)
  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mps = BaseMPS(tensors, backend=backend, center_position=2)
  gate = backend.convert_to_tensor(np.ones((2, 2, 2, 2), dtype=np.float64))
  with pytest.raises(ValueError):
    mps.apply_two_site_gate(gate=gate, site1=3, site2=4, max_singular_values=1)
  with pytest.raises(ValueError):
    mps.apply_two_site_gate(gate=gate, site1=3, site2=4, max_truncation_err=.1)


def test_apply_one_site_gate_2(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)
  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mps = BaseMPS(tensors, backend=backend, center_position=2)
  gate = backend.convert_to_tensor(np.array([[0, 1], [1, 0]], dtype=np.float64))
  mps.apply_one_site_gate(gate=gate, site=1)
  expected = np.array([[1., -2., 1.], [1., 2., 1.]])
  np.testing.assert_allclose(mps.tensors[1][0], expected)


def test_apply_one_site_gate_wrong_gate_raises_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)
  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mps = BaseMPS(tensors, backend=backend, center_position=2)
  gate1 = backend.convert_to_tensor(np.ones((2, 2, 2), dtype=np.float64))
  gate2 = backend.convert_to_tensor(np.ones((2, 2, 2), dtype=np.float64))
  with pytest.raises(ValueError):
    mps.apply_one_site_gate(gate=gate1, site=1)
  with pytest.raises(ValueError):
    mps.apply_one_site_gate(gate=gate2, site=1)


def test_apply_one_site_gate_invalid_site_raises_error(backend):
  backend = backend_factory.get_backend(backend)
  tensor = np.array([[[1., 2., 1.], [1., -2., 1.]],
                     [[-1., 1., -1.], [-1., 1., -1.]], [[1., 2, 3], [3, 2, 1]]],
                    dtype=np.float64)
  tensors = 6 * [backend.convert_to_tensor(tensor)]
  mps = BaseMPS(tensors, backend=backend, center_position=2)
  gate = backend.convert_to_tensor(np.ones((2, 2), dtype=np.float64))
  with pytest.raises(ValueError):
    mps.apply_one_site_gate(gate=gate, site=-1)
  with pytest.raises(ValueError):
    mps.apply_one_site_gate(gate=gate, site=6)
