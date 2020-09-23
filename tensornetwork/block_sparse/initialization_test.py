import numpy as np
import pytest
from tensornetwork.block_sparse.charge import (U1Charge, charge_equal,
                                               BaseCharge)
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import BlockSparseTensor
from tensornetwork.block_sparse.initialization import (zeros, ones, randn,
                                                       random, ones_like,
                                                       zeros_like, empty_like,
                                                       randn_like, random_like)

np_dtypes = [np.float64, np.complex128]


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_tn_zeros(dtype, num_charges):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(rank)
  ]
  arr = zeros(indices, dtype=dtype)
  np.testing.assert_allclose(arr.data, 0)
  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_tn_ones(dtype, num_charges):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(rank)
  ]

  arr = ones(indices, dtype=dtype)
  np.testing.assert_allclose(arr.data, 1)
  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_tn_random(dtype, num_charges):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(rank)
  ]
  arr = random(indices, dtype=dtype)

  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_tn_randn(dtype, num_charges):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(rank)
  ]
  arr = randn(indices, dtype=dtype)

  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('fun, val', [(ones_like, 1), (zeros_like, 0),
                                      (empty_like, None), (randn_like, None),
                                      (random_like, None)])
def test_like_init(fun, val, dtype, num_charges):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (Ds[n], num_charges)),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(rank)
  ]
  arr = randn(indices, dtype=dtype)
  arr2 = fun(arr)
  assert arr.dtype == arr2.dtype
  np.testing.assert_allclose(arr.shape, arr2.shape)
  np.testing.assert_allclose(arr.flat_flows, arr2.flat_flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], arr2.charges[n][0])
  if val is not None:
    np.testing.assert_allclose(arr2.data, val)
