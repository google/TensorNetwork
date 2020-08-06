import numpy as np
import pytest
import itertools
from tensornetwork.block_sparse.utils import (flatten, fuse_stride_arrays,
                                              fuse_ndarrays, fuse_degeneracies,
                                              _find_best_partition,
                                              _get_strides, intersect)


np_dtypes = [np.float64, np.complex128]
np_tensordot_dtypes = [np.float64, np.complex128]


def test_flatten():
  listoflist = [[1, 2], [3, 4], [5]]
  flat = flatten(listoflist)
  np.testing.assert_allclose(flat, [1, 2, 3, 4, 5])


def test_fuse_stride_arrays():
  dims = np.asarray([2, 3, 4, 5])
  strides = np.asarray([120, 60, 20, 5, 1])
  actual = fuse_stride_arrays(dims, strides)
  expected = fuse_ndarrays([
      np.arange(0, strides[n] * dims[n], strides[n], dtype=np.uint32)
      for n in range(len(dims))
  ])
  np.testing.assert_allclose(actual, expected)

  
def test_fuse_ndarrays():
  d1 = np.asarray([0, 1])
  d2 = np.asarray([2, 3, 4])
  fused = fuse_ndarrays([d1, d2])
  np.testing.assert_allclose(fused, [2, 3, 4, 3, 4, 5])


def test_fuse_degeneracies():
  d1 = np.asarray([0, 1])
  d2 = np.asarray([2, 3, 4])
  fused_degeneracies = fuse_degeneracies(d1, d2)
  np.testing.assert_allclose(fused_degeneracies, np.kron(d1, d2))

  
def test_find_best_partition():
  with pytest.raises(ValueError):
    _find_best_partition([5])


def test_find_best_partition_raises():
  d = [5, 4, 5, 2, 6, 8]
  p = _find_best_partition(d)
  assert p == 3

def test_intersect_1():
  a = np.array([[0, 1, 2], [2, 3, 4]])
  b = np.array([[0, -2, 6], [2, 3, 4]])
  out = intersect(a, b, axis=1)
  np.testing.assert_allclose(np.array([[0], [2]]), out)


def test_intersect_2():
  a = np.array([[0, 1, 2], [2, 3, 4]])
  b = np.array([[0, -2, 6, 2], [2, 3, 4, 4]])
  out, la, lb = intersect(a, b, axis=1, return_indices=True)
  np.testing.assert_allclose(np.array([[0, 2], [2, 4]]), out)
  np.testing.assert_allclose(la, [0, 2])
  np.testing.assert_allclose(lb, [0, 3])


def test_intersect_3():
  a = np.array([0, 1, 2, 3, 4])
  b = np.array([0, -1, 4])
  out = intersect(a, b)
  np.testing.assert_allclose([0, 4], out)


def test_intersect_4():
  a = np.array([0, 1, 2, 3, 4])
  b = np.array([0, -1, 4])
  out, la, lb = intersect(a, b, return_indices=True)
  np.testing.assert_allclose([0, 4], out)
  np.testing.assert_allclose(la, [0, 4])
  np.testing.assert_allclose(lb, [0, 2])


def test_intersect_raises():
  np.random.seed(10)
  a = np.random.randint(0, 10, (4, 5))
  b = np.random.randint(0, 10, (4, 6))
  with pytest.raises(ValueError):
    intersect(a, b, axis=0)
  c = np.random.randint(0, 10, (3, 7))
  with pytest.raises(ValueError):
    intersect(a, c, axis=1)
  with pytest.raises(NotImplementedError):
    intersect(a, c, axis=2)
  d = np.random.randint(0, 10, (3, 7, 3))
  e = np.random.randint(0, 10, (3, 7, 3))
  with pytest.raises(NotImplementedError):
    intersect(d, e, axis=1)
  
