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
import time
from tensornetwork.backends import backend_factory
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
from tensornetwork.ncon_interface import ncon

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


@pytest.mark.parametrize("N, pos", [(10, -1), (10, 10)])
def test_finite_mps_init_invalid_position_raises_value_error(backend, N, pos):
  D, d = 10, 2
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  with pytest.raises(ValueError):
    FiniteMPS(tensors, center_position=pos, backend=backend)


@pytest.mark.parametrize("N, pos", [(10, 0), (10, 9), (10, 5)])
def test_finite_mps_init(backend, N, pos):
  D, d = 10, 2
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  mps = FiniteMPS(tensors, center_position=pos, backend=backend)
  assert mps.center_position == pos


def test_canonical_finite_mps(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]
  D, d, N = 10, 2, 10
  tensors = [get_random_np((1, d, D), dtype)] + [
      get_random_np((D, d, D), dtype) for _ in range(N - 2)
  ] + [get_random_np((D, d, 1), dtype)]
  mps = FiniteMPS(
      tensors, center_position=N//2, backend=backend, canonicalize=True)
  mps.center_position += 1
  assert abs(mps.check_canonical()) > 1E-12
  mps.canonicalize()
  assert abs(mps.check_canonical()) < 1E-12


def test_local_measurement_finite_mps(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors_1 = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps_1 = FiniteMPS(tensors_1, center_position=0, backend=backend)

  tensors_2 = [np.zeros((1, d, D), dtype=dtype)] + [
      np.zeros((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.zeros((D, d, 1), dtype=dtype)]
  for t in tensors_2:
    t[0, 0, 0] = 1
  mps_2 = FiniteMPS(tensors_2, center_position=0, backend=backend)

  sz = np.diag([0.5, -0.5]).astype(dtype)
  result_1 = np.array(mps_1.measure_local_operator([sz] * N, range(N)))
  result_2 = np.array(mps_2.measure_local_operator([sz] * N, range(N)))
  np.testing.assert_almost_equal(result_1, np.zeros(N))
  np.testing.assert_allclose(result_2, np.ones(N) * 0.5)


@pytest.mark.parametrize("N1", [0, 5, 9])
def test_correlation_measurement_finite_mps(backend_dtype_values, N1):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors_1 = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps_1 = FiniteMPS(tensors_1, center_position=0, backend=backend)
  mps_1.position(N - 1)
  mps_1.position(0)
  tensors_2 = [np.zeros((1, d, D), dtype=dtype)] + [
      np.zeros((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.zeros((D, d, 1), dtype=dtype)]
  for t in tensors_2:
    t[0, 0, 0] = 1
  mps_2 = FiniteMPS(tensors_2, center_position=0, backend=backend)
  mps_2.position(N - 1)
  mps_2.position(0)

  sz = np.diag([0.5, -0.5]).astype(dtype)
  result_1 = np.array(mps_1.measure_two_body_correlator(sz, sz, N1, range(N)))
  result_2 = np.array(mps_2.measure_two_body_correlator(sz, sz, N1, range(N)))
  actual = np.zeros(N)
  actual[N1] = 0.25
  np.testing.assert_almost_equal(result_1, actual)
  np.testing.assert_allclose(result_2, np.ones(N) * 0.25)


def test_left_envs_one_site(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])
  envs = mps.left_envs(sites=[2])
  assert list(envs.keys()) == [2]
  expected = backend.convert_to_tensor(
      np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
  np.testing.assert_array_almost_equal(envs[2], expected)


def test_left_envs_one_site_center_position_to_right(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=4, backend=backend_dtype_values[0])
  envs = mps.left_envs(sites=[2])
  assert list(envs.keys()) == [2]
  np.testing.assert_array_almost_equal(envs[2], np.eye(3))


def test_left_envs_first_site(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])
  envs = mps.left_envs(sites=[0])
  assert list(envs.keys()) == [0]
  expected = 1.
  np.testing.assert_array_almost_equal(envs[0], expected)


def test_left_envs_last_site(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])
  envs = mps.left_envs(sites=[5])
  assert list(envs.keys()) == [5]
  expected = 1.
  np.testing.assert_array_almost_equal(envs[5], expected)


def test_left_envs_two_sites(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])
  envs = mps.left_envs(sites=[2, 3])
  assert list(envs.keys()) == [2, 3]
  expected = backend.convert_to_tensor(
      np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
  np.testing.assert_array_almost_equal(envs[2], expected)
  np.testing.assert_array_almost_equal(envs[3], expected)


def test_left_envs_two_non_consecutive_sites(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(
      tensors,
      center_position=None,
      backend=backend_dtype_values[0],
      canonicalize=False)
  l = backend.convert_to_tensor(np.ones((1, 1), dtype=dtype))
  exp = {}
  for n, t in enumerate(mps.tensors):
    if n in [1, 3]:
      exp[n] = l
    l = ncon([t, l, t], [[1, 2, -1], [1, 3], [3, 2, -2]], backend=backend)
  envs = mps.left_envs(sites=[1, 3])
  assert list(envs.keys()) == [1, 3]
  for n in [1, 3]:
    expected = exp[n]
    actual = envs[n]
    np.testing.assert_array_almost_equal(expected, actual)



def test_left_envs_two_non_consecutive_sites_2(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=4, backend=backend_dtype_values[0])
  envs = mps.left_envs(sites=[1, 3])
  assert list(envs.keys()) == [1, 3]
  np.testing.assert_array_almost_equal(envs[1], np.eye(2))
  np.testing.assert_array_almost_equal(envs[3], np.eye(3))


def test_left_envs_all_sites(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(
      tensors,
      center_position=N//2,
      backend=backend_dtype_values[0],
      canonicalize=True)
  l = backend.convert_to_tensor(np.ones((1, 1), dtype=dtype))
  exp = {}
  for n, t in enumerate(mps.tensors):
    exp[n] = l
    l = ncon([t, l, t], [[1, 2, -1], [1, 3], [3, 2, -2]], backend=backend)
  envs = mps.left_envs(sites=range(N))
  assert list(envs.keys()) == list(range(N))
  for n in range(N):
    expected = exp[n]
    actual = envs[n]
    np.testing.assert_array_almost_equal(expected, actual)



def test_left_envs_all_sites_non_0_center_position(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=2, backend=backend_dtype_values[0])
  envs = mps.left_envs(sites=[0, 1, 2, 3, 4, 5])
  assert list(envs.keys()) == [0, 1, 2, 3, 4, 5]
  expected = backend.convert_to_tensor(
      np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
  np.testing.assert_array_almost_equal(envs[0], 1.)
  np.testing.assert_array_almost_equal(envs[3], expected)


def test_left_envs_empty_seq(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])

  assert mps.left_envs(()) == {}
  assert mps.left_envs([]) == {}
  assert mps.left_envs(range(0)) == {}


def test_left_envs_invalid_sites_raises_error(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])
  with pytest.raises(ValueError):
    mps.left_envs(sites=[0, N + 1])
  with pytest.raises(ValueError):
    mps.left_envs(sites=[-1, N - 1])


def test_right_envs_one_site(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])
  envs = mps.right_envs(sites=[2])
  assert list(envs.keys()) == [2]
  np.testing.assert_array_almost_equal(envs[2], np.eye(3))


def test_right_envs_one_site_center_position_to_right(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=4, backend=backend_dtype_values[0])
  envs = mps.right_envs(sites=[2])
  assert list(envs.keys()) == [2]
  expected = backend.convert_to_tensor(
      np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
  np.testing.assert_array_almost_equal(envs[2], expected)


def test_right_envs_first_site(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])
  envs = mps.right_envs(sites=[-1])
  assert list(envs.keys()) == [-1]
  expected = 1.
  np.testing.assert_array_almost_equal(envs[-1], expected)


def test_right_envs_last_site(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])
  envs = mps.right_envs(sites=[4])
  assert list(envs.keys()) == [4]
  expected = 1.
  np.testing.assert_array_almost_equal(envs[4], expected)


def test_right_envs_two_sites(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])
  envs = mps.right_envs(sites=[2, 3])
  assert list(envs.keys()) == [2, 3]
  np.testing.assert_array_almost_equal(envs[2], np.eye(3))
  np.testing.assert_array_almost_equal(envs[3], np.eye(2))


def test_right_envs_two_non_consecutive_sites(backend_dtype_values):
  dtype = backend_dtype_values[1]
  backend = backend_factory.get_backend(backend_dtype_values[0])

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(
      tensors,
      center_position=None,
      backend=backend_dtype_values[0],
      canonicalize=False)
  r = backend.convert_to_tensor(np.ones((1, 1), dtype=dtype))
  exp = {}
  for n in reversed(range(N)):
    t = mps.tensors[n]
    if n in [1, 3]:
      exp[n] = r
    r = ncon([t, r, t], [[-1, 2, 1], [1, 3], [-2, 2, 3]], backend=backend)
  envs = mps.right_envs(sites=[1, 3])
  assert set(envs.keys()) == {3, 1}
  for n in [1, 3]:
    expected = exp[n]
    actual = envs[n]
    np.testing.assert_array_almost_equal(expected, actual)


def test_right_envs_two_non_consecutive_sites_2(backend_dtype_values):
  backend = backend_factory.get_backend(backend_dtype_values[0])
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=1, backend=backend_dtype_values[0])
  envs = mps.right_envs(sites=[1, 3])
  assert set(envs.keys()) == {1, 3}
  exp1 = backend.convert_to_tensor(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
  exp3 = backend.convert_to_tensor(np.array([[1, 0], [0, 1]]))

  np.testing.assert_array_almost_equal(envs[1], exp1)
  np.testing.assert_array_almost_equal(envs[3], exp3)


def test_right_envs_all_sites(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])
  envs = mps.right_envs(sites=[-1, 0, 1, 2, 3, 4])
  assert set(envs.keys()) == {-1, 0, 1, 2, 3, 4}
  np.testing.assert_array_almost_equal(envs[-1], 1.)
  np.testing.assert_array_almost_equal(envs[2], np.eye(3))


def test_right_envs_all_sites_non_0_center_position(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 3, 2, 5
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=2, backend=backend_dtype_values[0])
  envs = mps.right_envs(sites=[-1, 0, 1, 2, 3, 4])
  assert set(envs.keys()) == {-1, 0, 1, 2, 3, 4}
  np.testing.assert_array_almost_equal(envs[-1], 1.)
  np.testing.assert_array_almost_equal(envs[2], np.eye(3))


def test_right_envs_empty_seq(backend_dtype_values):
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend_dtype_values[0])

  assert mps.right_envs(()) == {}
  assert mps.right_envs([]) == {}
  assert mps.right_envs(range(0)) == {}


def test_right_envs_invalid_sites_raises_error(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend)
  with pytest.raises(ValueError):
    mps.right_envs(sites=[-1, N])
  with pytest.raises(ValueError):
    mps.right_envs(sites=[-2, N - 1])


def test_random_mps(backend_dtype_values):
  mps = FiniteMPS.random(
      d=[3, 4, 5],
      D=[2, 3],
      dtype=backend_dtype_values[1],
      backend=backend_dtype_values[0])
  assert len(mps) == 3
  assert mps.physical_dimensions == [3, 4, 5]
  assert mps.bond_dimensions == [1, 2, 3, 1]


def test_random_mps_invalid_dimensions_raises_error(backend_dtype_values):
  with pytest.raises(ValueError):
    FiniteMPS.random(
        d=[3, 4],
        D=[2, 3],
        dtype=backend_dtype_values[1],
        backend=backend_dtype_values[0])
  with pytest.raises(ValueError):
    FiniteMPS.random(
        d=[3, 4, 4, 2],
        D=[2, 3],
        dtype=backend_dtype_values[1],
        backend=backend_dtype_values[0])


def test_save_not_implemented(backend_dtype_values):
  backend = backend_dtype_values[0]
  dtype = backend_dtype_values[1]

  D, d, N = 1, 2, 10
  tensors = [np.ones((1, d, D), dtype=dtype)] + [
      np.ones((D, d, D), dtype=dtype) for _ in range(N - 2)
  ] + [np.ones((D, d, 1), dtype=dtype)]
  mps = FiniteMPS(tensors, center_position=0, backend=backend)
  with pytest.raises(NotImplementedError):
    mps.save('tmp')


def test_check_canonical_raises(backend):
  N, D, d = 10, 10, 2
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  mps = FiniteMPS(
      tensors, center_position=None, canonicalize=False, backend=backend)
  with pytest.raises(
      ValueError,
      match="FiniteMPS.center_positions is `None`. "
      "Cannot check canonical form."):
    mps.check_canonical()
