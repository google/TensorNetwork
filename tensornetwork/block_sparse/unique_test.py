import numpy as np
import pytest
from tensornetwork.block_sparse.unique import unique, get_dtype, intersect, collapse, expand


def get_index(return_index, return_inverse, return_counts, which):
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
                                                (4, np.int16, np.int64)])
def test_collapse(N, dtype, resdtype):
  D = 10000
  a = np.random.randint(-5, 5, (D, N), dtype=dtype)
  collapsed = collapse(a)
  if N == 3:
    expected = np.squeeze(
        np.concatenate([a, np.zeros((D, 1), dtype=dtype)],
                       axis=1).view(resdtype))
  else:
    expected = np.squeeze(a.view(resdtype))
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
  if N == 3:
    expected = np.concatenate([expected, np.zeros((D, 1), dtype=dtype)], axis=1)
  actual = expand(collapsed, dtype)
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
  actual = unique(a, return_index, return_inverse, return_counts, label_dtype)

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


# def test_intersect(return_indices, dtype, N, D):
#   a = np.random.randint(-10, 10, (D, N), dtype=dtype)
#   b = np.random.randint(-10, 10, (D, N), dtype=dtype)
#   expected = np.intersect1d(a, b, return_indices)
#   actual = intersect(a, b, return_indices)


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

def test_intersect_5():
  a = np.array([[0, 2], [1, 3], [2, 4]])
  b = np.array([[0, 2], [-2, 3], [6, 6]])
  out = intersect(a, b, axis=0)
  np.testing.assert_allclose(np.array([[0, 2]]), out)

def test_intersect_6():
  a = np.array([[0,2], [1,3], [2,4]])
  b = np.array([[0,2], [-2, 3], [6, 4], [2, 4]])
  out, la, lb = intersect(a, b, axis=0, return_indices=True)
  np.testing.assert_allclose(np.array([[0,2], [2,4]]), out)
  np.testing.assert_allclose(la, [0, 2])
  np.testing.assert_allclose(lb, [0, 3])

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
