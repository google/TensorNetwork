import numpy as np
import pytest
import itertools
from tensornetwork.block_sparse.utils import (
    flatten, fuse_stride_arrays, fuse_ndarrays, fuse_degeneracies,
    _find_best_partition, _get_strides, unique, get_dtype, get_real_dtype,
    intersect, collapse, expand, _intersect_ndarray)

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

#pylint: disable=too-many-return-statements
def get_index(return_index, return_inverse, return_counts, which):#pylint: disable=inconsistent-return-statements
  if which == 'index':
    return 1 if return_index else -1
  if which == 'inverse':
    if return_index:
      return 2 if return_inverse else -1
    return 1 if return_inverse else -1
  if which == 'counts':
    if return_index and return_inverse:
      return 3 if return_counts else -1
    if return_index or return_inverse:
      return 2 if return_counts else -1
    return 1 if return_counts else -1


@pytest.mark.parametrize('N, dtype, resdtype', [(1, np.int8, np.int8),
                                                (2, np.int8, np.int16),
                                                (2, np.int16, np.int32),
                                                (2, np.int32, np.int64),
                                                (3, np.int8, np.int32),
                                                (3, np.int16, np.int64),
                                                (4, np.int8, np.int32),
                                                (4, np.int16, np.int64),
                                                (5, np.int8, np.int64),
                                                (6, np.int8, np.int64)])
def test_collapse(N, dtype, resdtype):
  D = 10000
  a = np.random.randint(-5, 5, (D, N), dtype=dtype)
  collapsed = collapse(a)
  if N in (1, 2, 4):
    expected = np.squeeze(a.view(resdtype))
  elif N == 3:
    expected = np.squeeze(
        np.concatenate([a, np.zeros((D, 1), dtype=dtype)],
                       axis=1).view(resdtype))
  elif N > 4:
    expected = np.squeeze(
        np.concatenate([a, np.zeros((D, 8 - N), dtype=dtype)],
                       axis=1).view(resdtype))

  np.testing.assert_allclose(collapsed, expected)


def test_collapse_2():
  N = 2
  dtype = np.int64
  D = 10000
  a = np.random.randint(-5, 5, (D, N), dtype=dtype)
  collapsed = collapse(a)
  np.testing.assert_allclose(collapsed, a)


@pytest.mark.parametrize('N, dtype',
                         [(1, np.int8), (1, np.int16), (1, np.int32),
                          (1, np.int64), (2, np.int8), (2, np.int16),
                          (2, np.int32), (3, np.int8), (3, np.int16),
                          (4, np.int8), (4, np.int16), (4, np.int32)])
def test_collapse_expand(N, dtype):
  D = 10000
  expected = np.random.randint(-5, 5, (D, N), dtype=dtype)
  collapsed = collapse(expected)
  actual = expand(collapsed, dtype, original_width=N, original_ndim=2)
  np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize('N, label_dtype, D', [(1, np.int8, 1000),
                                               (1, np.int16, 1000),
                                               (2, np.int8, 100),
                                               (2, np.int16, 1000),
                                               (3, np.int8, 100),
                                               (3, np.int16, 10000),
                                               (4, np.int8, 100),
                                               (4, np.int16, 10000),
                                               (4, np.int32, 100000)])
@pytest.mark.parametrize('return_index', [True, False])
@pytest.mark.parametrize('return_inverse', [True, False])
@pytest.mark.parametrize('return_counts', [True, False])
def test_unique(N, label_dtype, D, return_index, return_inverse, return_counts):
  a = np.random.randint(-10, 10, (D, N), dtype=np.int16)
  expected = np.unique(a, return_index, return_inverse, return_counts, axis=0)
  actual = unique(
      a, return_index, return_inverse, return_counts, label_dtype=label_dtype)

  def test_array(act, exp):
    if N != 3:
      t1 = np.squeeze(exp.view(get_dtype(2 * N)))
      t2 = np.squeeze(act.view(get_dtype(2 * N)))
      ordact = np.argsort(t2)
      ordexp = np.argsort(t1)
    else:
      t1 = np.squeeze(
          np.concatenate([exp, np.zeros((exp.shape[0], 1), dtype=np.int16)],
                         axis=1).view(get_dtype(2 * N)))

      t2 = np.squeeze(
          np.concatenate([act, np.zeros((act.shape[0], 1), dtype=np.int16)],
                         axis=1).view(get_dtype(2 * N)))

      ordact = np.argsort(t2)
      ordexp = np.argsort(t1)
    np.testing.assert_allclose(t1[ordexp], t2[ordact])
    return ordact, ordexp

  if not any([return_index, return_inverse, return_counts]):
    test_array(actual, expected)
  else:
    ordact, ordexp = test_array(actual[0], expected[0])
    if return_index:
      ind = get_index(return_index, return_inverse, return_counts, 'index')
      np.testing.assert_allclose(actual[ind][ordact], expected[ind][ordexp])
    if return_inverse:
      ind = get_index(return_index, return_inverse, return_counts, 'inverse')
      mapact = np.zeros(len(ordact), dtype=np.int16)
      mapact[ordact] = np.arange(len(ordact), dtype=np.int16)
      mapexp = np.zeros(len(ordexp), dtype=np.int16)
      mapexp[ordexp] = np.arange(len(ordexp), dtype=np.int16)
      np.testing.assert_allclose(mapact[actual[ind]], mapexp[expected[ind]])
      assert actual[ind].dtype == label_dtype
    if return_counts:
      ind = get_index(return_index, return_inverse, return_counts, 'counts')
      np.testing.assert_allclose(actual[ind][ordact], expected[ind][ordexp])


@pytest.mark.parametrize('return_index', [True, False])
@pytest.mark.parametrize('return_inverse', [True, False])
@pytest.mark.parametrize('return_counts', [True, False])
def test_unique_2(return_index, return_inverse, return_counts):
  N = 5
  D = 1000
  a = np.random.randint(-10, 10, (D, N), dtype=np.int16)
  expected = np.unique(a, return_index, return_inverse, return_counts, axis=0)
  actual = unique(a, return_index, return_inverse, return_counts)
  if not any([return_index, return_inverse, return_counts]):
    np.testing.assert_allclose(expected, actual)
  else:
    for n, e in enumerate(expected):
      np.testing.assert_allclose(e, actual[n])

@pytest.mark.parametrize('return_index', [True, False])
@pytest.mark.parametrize('return_inverse', [True, False])
@pytest.mark.parametrize('return_counts', [True, False])
def test_unique_1d(return_index, return_inverse, return_counts):
  D = 1000
  a = np.random.randint(-10, 10, D, dtype=np.int16)
  expected = np.unique(a, return_index, return_inverse, return_counts)
  actual = unique(a, return_index, return_inverse, return_counts)
  if not any([return_index, return_inverse, return_counts]):
    np.testing.assert_allclose(expected, actual)
  else:
    for n, e in enumerate(expected):
      np.testing.assert_allclose(e, actual[n])

@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
def test_intersect_1(dtype):
  a = np.array([[0, 1, 2], [2, 3, 4]], dtype=dtype)
  b = np.array([[0, -2, 6], [2, 3, 4]], dtype=dtype)
  out = intersect(a, b, axis=1)
  np.testing.assert_allclose(np.array([[0], [2]]), out)

@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
def test_intersect_2(dtype):
  a = np.array([[0, 1, 2], [2, 3, 4]], dtype=dtype)
  b = np.array([[0, -2, 6, 2], [2, 3, 4, 4]], dtype=dtype)
  out, la, lb = intersect(a, b, axis=1, return_indices=True)
  np.testing.assert_allclose(np.array([[0, 2], [2, 4]]), out)
  np.testing.assert_allclose(la, [0, 2])
  np.testing.assert_allclose(lb, [0, 3])

@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
def test_intersect_3(dtype):
  a = np.array([0, 1, 2, 3, 4], dtype=dtype)
  b = np.array([0, -1, 4], dtype=dtype)
  out = intersect(a, b)
  np.testing.assert_allclose([0, 4], out)

@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
def test_intersect_4(dtype):
  a = np.array([0, 1, 2, 3, 4], dtype=dtype)
  b = np.array([0, -1, 4], dtype=dtype)
  out, la, lb = intersect(a, b, return_indices=True)
  np.testing.assert_allclose([0, 4], out)
  np.testing.assert_allclose(la, [0, 4])
  np.testing.assert_allclose(lb, [0, 2])

def test_intersect_raises():
  np.random.seed(10)
  a = np.random.randint(0, 10, (4, 5, 1))
  b = np.random.randint(0, 10, (4, 6))
  with pytest.raises(ValueError, match="array ndims"):
    intersect(a, b, axis=0)
  a = np.random.randint(0, 10, (4, 5))
  b = np.random.randint(0, 10, (4, 6))
  with pytest.raises(ValueError, match="array widths"):
    intersect(a, b, axis=0)
  c = np.random.randint(0, 10, (3, 7))
  with pytest.raises(ValueError, match="array heights"):
    intersect(a, c, axis=1)
  with pytest.raises(NotImplementedError, match="intersection can only"):
    intersect(a, c, axis=2)
  d = np.random.randint(0, 10, (3, 7, 3))
  e = np.random.randint(0, 10, (3, 7, 3))
  with pytest.raises(NotImplementedError, match="_intersect_ndarray is only"):
    intersect(d, e, axis=1)
  a = np.random.randint(0, 10, (4, 5), dtype=np.int16)
  b = np.random.randint(0, 10, (4, 6), dtype=np.int32)
  with pytest.raises(ValueError, match="array dtypes"):
    intersect(a, b, axis=0)

@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
def test_intersect_5(dtype):
  a = np.array([[0, 2], [1, 3], [2, 4]], dtype=dtype)
  b = np.array([[0, 2], [-2, 3], [6, 6]], dtype=dtype)
  out = intersect(a, b, axis=0)
  np.testing.assert_allclose(np.array([[0, 2]]), out)

@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
def test_intersect_6(dtype):
  a = np.array([[0, 2], [1, 3], [2, 4]], dtype=dtype)
  b = np.array([[0, 2], [-2, 3], [6, 4], [2, 4]], dtype=dtype)
  out, la, lb = intersect(a, b, axis=0, return_indices=True)
  np.testing.assert_allclose(np.array([[0, 2], [2, 4]]), out)
  np.testing.assert_allclose(la, [0, 2])
  np.testing.assert_allclose(lb, [0, 3])


@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64])
def test_intersect_1d(dtype):
  a = np.random.randint(-5, 5, 10, dtype=dtype)
  b = np.random.randint(-2, 2, 8, dtype=dtype)
  out, la, lb = intersect(a, b, axis=0, return_indices=True)
  out_, la_, lb_ = np.intersect1d(a, b, return_indices=True)
  np.testing.assert_allclose(out, out_)
  np.testing.assert_allclose(la, la_)
  np.testing.assert_allclose(lb, lb_)

def test_intersect_ndarray_1():
  a = np.array([[0, 1, 2], [2, 3, 4]])
  b = np.array([[0, -2, 6], [2, 3, 4]])
  out = _intersect_ndarray(a, b, axis=1)
  np.testing.assert_allclose(np.array([[0], [2]]), out)


def test_intersect_ndarray_2():
  a = np.array([[0, 1, 2], [2, 3, 4]])
  b = np.array([[0, -2, 6, 2], [2, 3, 4, 4]])
  out, la, lb = _intersect_ndarray(a, b, axis=1, return_indices=True)
  np.testing.assert_allclose(np.array([[0, 2], [2, 4]]), out)
  np.testing.assert_allclose(la, [0, 2])
  np.testing.assert_allclose(lb, [0, 3])


def test_intersect_ndarray_3():
  a = np.array([0, 1, 2, 3, 4])
  b = np.array([0, -1, 4])
  out = _intersect_ndarray(a, b)
  np.testing.assert_allclose([0, 4], out)


def test_intersect_ndarray_4():
  a = np.array([0, 1, 2, 3, 4])
  b = np.array([0, -1, 4])
  out, la, lb = _intersect_ndarray(a, b, return_indices=True)
  np.testing.assert_allclose([0, 4], out)
  np.testing.assert_allclose(la, [0, 4])
  np.testing.assert_allclose(lb, [0, 2])


def test_intersect_ndarray_5():
  a = np.array([[0, 2], [1, 3], [2, 4]])
  b = np.array([[0, 2], [-2, 3], [6, 6]])
  out = _intersect_ndarray(a, b, axis=0)
  np.testing.assert_allclose(np.array([[0, 2]]), out)

def test_intersect_ndarray_6():
  a = np.array([[0, 2], [1, 3], [2, 4]])
  b = np.array([[0, 2], [-2, 3], [6, 4], [2, 4]])
  out, la, lb = _intersect_ndarray(a, b, axis=0, return_indices=True)
  np.testing.assert_allclose(np.array([[0, 2], [2, 4]]), out)
  np.testing.assert_allclose(la, [0, 2])
  np.testing.assert_allclose(lb, [0, 3])

def test_intersect_ndarray_1d():
  a = np.random.randint(-5, 5, 10)
  b = np.random.randint(-2, 2, 8)
  out, la, lb = _intersect_ndarray(a, b, axis=0, return_indices=True)
  out_, la_, lb_ = np.intersect1d(a, b, return_indices=True)
  np.testing.assert_allclose(out, out_)
  np.testing.assert_allclose(la, la_)
  np.testing.assert_allclose(lb, lb_)

def test_intersect_ndarray_raises():
  np.random.seed(10)
  a = np.random.randint(0, 10, (4, 5, 1))
  b = np.random.randint(0, 10, (4, 6))
  with pytest.raises(ValueError, match="array ndims"):
    _intersect_ndarray(a, b, axis=0)
  a = np.random.randint(0, 10, (4, 5))
  b = np.random.randint(0, 10, (4, 6))
  with pytest.raises(ValueError, match="array widths"):
    _intersect_ndarray(a, b, axis=0)
  with pytest.raises(NotImplementedError, match="intersection can only"):
    _intersect_ndarray(a, b, axis=2)
  d = np.random.randint(0, 10, (3, 7, 3))
  e = np.random.randint(0, 10, (3, 7, 3))
  with pytest.raises(NotImplementedError, match="_intersect_ndarray is only"):
    _intersect_ndarray(d, e, axis=1)

def test_get_real_dtype():
  assert get_real_dtype(np.complex128) == np.float64
  assert get_real_dtype(np.complex64) == np.float32
  assert get_real_dtype(np.float64) == np.float64
  assert get_real_dtype(np.float32) == np.float32


def test_get_dtype():
  assert get_dtype(1) == np.int8
  assert get_dtype(2) == np.int16
  assert get_dtype(3) == np.int32
  assert get_dtype(4) == np.int32
  assert get_dtype(5) == np.int64
  assert get_dtype(6) == np.int64
  assert get_dtype(7) == np.int64
  assert get_dtype(8) == np.int64
  assert get_dtype(9) == np.int64
