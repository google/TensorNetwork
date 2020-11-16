import numpy as np
import pytest
from tensornetwork.block_sparse.charge import (U1Charge, fuse_charges,
                                               charge_equal,
                                               fuse_ndarray_charges, BaseCharge,
                                               Z2Charge)
from tensornetwork.block_sparse.utils import fuse_ndarrays
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import (ChargeArray,
                                                          BlockSparseTensor,
                                                          compare_shapes,
                                                          tensordot)

np_dtypes = [np.float64, np.complex128]

def get_charge(chargetype, num_charges, D):
  if chargetype == "U1":
    out = BaseCharge(
        np.random.randint(-5, 6, (D, num_charges)),
        charge_types=[U1Charge] * num_charges)
  if chargetype == "Z2":
    out = BaseCharge(
        np.random.randint(0, 2, (D, num_charges)),
        charge_types=[Z2Charge] * num_charges)
  if chargetype == "mixed":
    n1 = num_charges // 2 if num_charges > 1 else 1
    out = BaseCharge(
        np.random.randint(-5, 6, (D, n1)), charge_types=[U1Charge] * n1)

    if num_charges > 1:
      n2 = num_charges - n1
      out = out @ BaseCharge(
          np.random.randint(0, 2, (D, n2)), charge_types=[Z2Charge] * n2)

  return out

@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
def test_ChargeArray_init(chargetype):
  np.random.seed(10)
  D = 10
  rank = 4
  charges = [get_charge(chargetype, 1, D) for _ in range(rank)]
  data = np.random.uniform(0, 1, size=D**rank)
  flows = np.random.choice([True, False], size=rank, replace=True)
  order = [[n] for n in range(rank)]
  arr = ChargeArray(data, charges, flows, order=order)
  np.testing.assert_allclose(data, arr.data)
  for c1, c2 in zip(charges, arr.charges):
    assert charge_equal(c1, c2[0])
  for c1, c2 in zip(charges, arr._charges):
    assert charge_equal(c1, c2)


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
def test_ChargeArray_init_raises(chargetype):
  np.random.seed(10)
  D = 10
  rank = 4
  charges = [get_charge(chargetype, 1, D) for _ in range(rank)]
  data = np.random.uniform(0, 1, size=D**rank)
  flows = np.random.choice([True, False], size=rank, replace=True)
  order = [[n + 10] for n in range(rank)]
  with pytest.raises(ValueError):
    ChargeArray(data, charges, flows, order=order, check_consistency=True)


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('dtype', np_dtypes)
def test_ChargeArray_generic(dtype, chargetype):
  Ds = [8, 9, 10, 11]
  indices = [Index(get_charge(chargetype, 1, Ds[n]), False) for n in range(4)]
  arr = ChargeArray.random(indices, dtype=dtype)
  assert arr.ndim == 4
  assert arr.dtype == dtype
  np.testing.assert_allclose(arr.shape, Ds)
  np.testing.assert_allclose(arr.flat_flows, [False, False, False, False])
  for n in range(4):
    assert charge_equal(indices[n]._charges[0], arr.flat_charges[n])
    assert arr.sparse_shape[n] == indices[n]


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
def test_ChargeArray_todense(dtype, num_charges, chargetype):
  Ds = [8, 9, 10, 11]
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), False) for n in range(4)
  ]
  arr = ChargeArray.random(indices, dtype=dtype)
  np.testing.assert_allclose(arr.todense(), np.reshape(arr.data, Ds))


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize(
    'Ds', [[[10, 12], [11]], [[8, 9], [10, 11]], [[8, 9], [10, 11], [12, 13]]])
@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
def test_ChargeArray_reshape(dtype, Ds, chargetype):
  flat_Ds = sum(Ds, [])
  R = len(flat_Ds)
  indices = [
      Index(get_charge(chargetype, 1, flat_Ds[n]), False) for n in range(R)
  ]
  arr = ChargeArray.random(indices, dtype=dtype)

  ds = [np.prod(D) for D in Ds]
  arr2 = arr.reshape(ds)
  cnt = 0
  for n in range(arr2.ndim):
    for m in range(len(arr2.charges[n])):
      assert charge_equal(arr2.charges[n][m], indices[cnt].charges)
      cnt += 1
  order = []
  flows = []
  start = 0
  for D in Ds:
    order.append(list(range(start, start + len(D))))
    start += len(D)
    flows.append([False] * len(D))

  np.testing.assert_allclose(arr2.shape, ds)
  for n in range(len(arr2._order)):
    np.testing.assert_allclose(arr2._order[n], order[n])
    np.testing.assert_allclose(arr2.flows[n], flows[n])
  assert arr2.ndim == len(Ds)
  arr3 = arr.reshape(flat_Ds)
  for n in range(len(Ds)):
    assert charge_equal(arr3.charges[n][0], indices[n].charges)

  np.testing.assert_allclose(arr3.shape, flat_Ds)
  np.testing.assert_allclose(arr3._order, [[n] for n in range(len(flat_Ds))])
  np.testing.assert_allclose(arr3.flows, [[False] for n in range(len(flat_Ds))])
  assert arr3.ndim == len(flat_Ds)

@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
def test_ChargeArray_reshape_with_index(dtype, chargetype):
  Ds = [8, 9, 10, 11]
  R = len(Ds)
  indices = [
      Index(get_charge(chargetype, 1, Ds[n]), False) for n in range(R)
  ]
  arr = ChargeArray.random(indices, dtype=dtype)
  arr2 = arr.reshape([indices[0] * indices[1], indices[2] * indices[3]])
  cnt = 0
  for n in range(arr2.ndim):
    for m in range(len(arr2.charges[n])):
      assert charge_equal(arr2.charges[n][m], indices[cnt].charges)
      cnt += 1
  np.testing.assert_allclose(arr2.shape, [72, 110])
  assert arr2.ndim == 2


def test_ChargeArray_reshape_raises():
  Ds = [8, 9, 10, 11]
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), False)
      for n in range(4)
  ]
  arr = ChargeArray.random(indices)

  with pytest.raises(ValueError, match=r"The shape \(2, 4, 9, 2, 5, 11\)"):
    arr.reshape([2, 4, 9, 2, 5, 11])

  with pytest.raises(ValueError, match="A tensor with"):
    arr.reshape([64, 65])

  arr2 = arr.reshape([72, 110])
  with pytest.raises(
      ValueError,
      match=r"The shape \(9, 8, 10, 11\) is incompatible with the"
      r" elementary shape \(8, 9, 10, 11\) of the tensor."):

    arr2.reshape([9, 8, 10, 11])

  Ds = [8, 9, 0, 11]
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), False)
      for n in range(4)
  ]
  arr3 = ChargeArray.random(indices)
  with pytest.raises(ValueError):
    arr3.reshape([72, 0])


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
def test_ChargeArray_transpose(chargetype):
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(get_charge(chargetype, 1, Ds[n]), flows[n]) for n in range(4)
  ]
  arr = ChargeArray.random(indices)
  order = [2, 1, 0, 3]
  arr2 = arr.transpose(order)
  np.testing.assert_allclose(Ds[order], arr2.shape)
  np.testing.assert_allclose(arr2._order, [[2], [1], [0], [3]])
  np.testing.assert_allclose(arr2.flows, [[True], [False], [True], [False]])

@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_ChargeArray_transpose_shuffle(chargetype, num_charges):
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  order = [2, 0, 1, 3]
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), flows[n])
      for n in range(4)
  ]
  arr = ChargeArray.random(indices)
  data = np.ascontiguousarray(np.transpose(np.reshape(arr.data, Ds), order))
  arr2 = arr.transpose(order, shuffle=True)
  data3 = np.reshape(arr2.data, Ds[order])
  np.testing.assert_allclose(data, data3)
  np.testing.assert_allclose(arr2.shape, Ds[order])
  np.testing.assert_allclose(arr2._order, [[0], [1], [2], [3]])
  np.testing.assert_allclose(arr2.flows, [[True], [True], [False], [False]])


def test_ChargeArray_transpose_raises():
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), flows[n])
      for n in range(4)
  ]
  arr = ChargeArray.random(indices)
  order = [2, 1, 0]
  with pytest.raises(ValueError):
    arr.transpose(order)


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
def test_ChargeArray_transpose_reshape(chargetype):
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(get_charge(chargetype, 1, Ds[n]), flows[n]) for n in range(4)
  ]
  arr = ChargeArray.random(indices)
  arr2 = arr.transpose([2, 0, 1, 3])
  arr3 = arr2.reshape([80, 99])
  np.testing.assert_allclose(arr3.shape, [80, 99])
  np.testing.assert_allclose(arr3._order, [[2, 0], [1, 3]])
  np.testing.assert_allclose(arr3.flows, [[True, True], [False, False]])

  arr4 = arr3.transpose([1, 0])
  np.testing.assert_allclose(arr4.shape, [99, 80])
  np.testing.assert_allclose(arr4._order, [[1, 3], [2, 0]])
  np.testing.assert_allclose(arr4.flows, [[False, False], [True, True]])

  arr5 = arr4.reshape([9, 11, 10, 8])
  np.testing.assert_allclose(arr5.shape, [9, 11, 10, 8])
  np.testing.assert_allclose(arr5._order, [[1], [3], [2], [0]])
  np.testing.assert_allclose(arr5.flows, [[False], [False], [True], [True]])


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_ChargeArray_contiguous(num_charges, chargetype):
  Ds = np.array([8, 9, 10, 11])
  order = [2, 0, 1, 3]
  flows = [True, False, True, False]
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), flows[n])
      for n in range(4)
  ]
  arr = ChargeArray.random(indices)
  data = np.ascontiguousarray(np.transpose(np.reshape(arr.data, Ds), order))
  arr2 = arr.transpose(order).contiguous()
  data3 = np.reshape(arr2.data, Ds[order])
  np.testing.assert_allclose(data, data3)
  np.testing.assert_allclose(arr2.shape, Ds[order])
  np.testing.assert_allclose(arr2._order, [[0], [1], [2], [3]])
  np.testing.assert_allclose(arr2.flows, [[True], [True], [False], [False]])


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_ChargeArray_transpose_reshape_contiguous(num_charges, chargetype):
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), flows[n])
      for n in range(4)
  ]
  arr = ChargeArray.random(indices)
  nparr = np.reshape(arr.data, Ds)

  arr2 = arr.transpose([2, 0, 1, 3])
  nparr2 = nparr.transpose([2, 0, 1, 3])
  arr3 = arr2.reshape([80, 99])
  nparr3 = nparr2.reshape([80, 99])
  arr4 = arr3.transpose([1, 0])
  nparr4 = nparr3.transpose([1, 0])

  arr5 = arr4.reshape([9, 11, 10, 8])
  nparr5 = nparr4.reshape([9, 11, 10, 8])
  np.testing.assert_allclose(arr3.contiguous().data,
                             np.ascontiguousarray(nparr3).flat)
  np.testing.assert_allclose(arr4.contiguous().data,
                             np.ascontiguousarray(nparr4).flat)
  np.testing.assert_allclose(arr5.contiguous().data,
                             np.ascontiguousarray(nparr5).flat)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_ChargeArray_conj(dtype):
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), flows[n])
      for n in range(4)
  ]
  arr = ChargeArray.random(indices, dtype=dtype)
  conj = arr.conj()
  np.testing.assert_allclose(conj.data, np.conj(arr.data))


def test_BlockSparseTensor_init():
  np.random.seed(10)
  D = 10
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  charges = [
      U1Charge.random(dimension=D, minval=-5, maxval=5) for _ in range(rank)
  ]
  fused = fuse_charges(charges, flows)
  data = np.random.uniform(
      0, 1, size=len(np.nonzero(fused == np.zeros((1, 1)))[0]))
  order = [[n] for n in range(rank)]
  arr = BlockSparseTensor(data, charges, flows, order=order)
  np.testing.assert_allclose(data, arr.data)
  for c1, c2 in zip(charges, arr.charges):
    assert charge_equal(c1, c2[0])
  for c1, c2 in zip(charges, arr._charges):
    assert charge_equal(c1, c2)
  data = np.random.uniform(
      0, 1, size=len(np.nonzero(fused == np.zeros((1, 1)))[0]) + 1)
  with pytest.raises(ValueError):
    arr = BlockSparseTensor(
        data, charges, flows, order=order, check_consistency=True)


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_BlockSparseTensor_random(dtype, num_charges, chargetype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), flows[n])
      for n in range(rank)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)

  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_BlockSparseTensor_randn(dtype, num_charges, chargetype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), flows[n])
      for n in range(rank)
  ]
  arr = BlockSparseTensor.randn(indices, dtype=dtype)

  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_BlockSparseTensor_ones(dtype, num_charges, chargetype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), flows[n])
      for n in range(rank)
  ]
  arr = BlockSparseTensor.ones(indices, dtype=dtype)
  np.testing.assert_allclose(arr.data, 1)
  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_BlockSparseTensor_zeros(dtype, num_charges, chargetype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), flows[n])
      for n in range(rank)
  ]
  arr = BlockSparseTensor.zeros(indices, dtype=dtype)
  np.testing.assert_allclose(arr.data, 0)
  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
def test_BlockSparseTensor_copy(chargetype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(get_charge(chargetype, 1, Ds[n]), flows[n]) for n in range(rank)
  ]
  arr = BlockSparseTensor.randn(indices)
  copy = arr.copy()
  assert arr.data is not copy.data
  for n in range(len(arr._charges)):
    assert arr._charges[n] is not copy._charges[n]
  assert arr._flows is not copy._flows
  assert arr._order is not copy._order


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_todense(num_charges, chargetype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  charges = [get_charge(chargetype, num_charges, Ds[n]) for n in range(rank)]
  fused = fuse_charges(charges, flows)
  mask = fused == np.zeros((1, num_charges))
  inds = np.nonzero(mask)[0]
  inds2 = np.nonzero(np.logical_not(mask))[0]
  indices = [Index(charges[n], flows[n]) for n in range(rank)]
  arr = BlockSparseTensor.randn(indices)
  dense = np.array(arr.todense().flat)
  np.testing.assert_allclose(dense[inds], arr.data)
  np.testing.assert_allclose(dense[inds2], 0)


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_fromdense(num_charges, chargetype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  charges = [get_charge(chargetype, num_charges, Ds[n]) for n in range(rank)]
  fused = fuse_charges(charges, flows)
  mask = fused == np.zeros((1, num_charges))
  inds = np.nonzero(mask)[0]
  inds2 = np.nonzero(np.logical_not(mask))[0]
  indices = [Index(charges[n], flows[n]) for n in range(rank)]

  dense = np.random.random_sample(Ds)
  arr = BlockSparseTensor.fromdense(indices, dense)
  dense_arr = arr.todense()

  np.testing.assert_allclose(np.ravel(dense)[inds], arr.data)
  np.testing.assert_allclose(np.ravel(dense_arr)[inds2], 0)


def test_fromdense_raises():
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = len(Ds)
  flows = np.random.choice([True, False], size=rank, replace=True)
  charges = [U1Charge.random(Ds[n], -5, 5) for n in range(rank)]
  indices = [Index(charges[n], flows[n]) for n in range(rank)]

  dense = np.random.random_sample([8, 9, 9, 11])
  with pytest.raises(ValueError):
    _ = BlockSparseTensor.fromdense(indices, dense)


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('op', [np.add, np.subtract])
@pytest.mark.parametrize('dtype', np_dtypes)
def test_add_sub(op, dtype, num_charges, chargetype):
  np.random.seed(10)
  indices = [
      Index(get_charge(chargetype, num_charges, 10), False) for _ in range(4)
  ]
  order = np.arange(4)
  np.random.shuffle(order)
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = BlockSparseTensor.randn([indices[o] for o in order], dtype=dtype)
  npb = np.reshape(b.todense(), b.shape)
  npa = np.reshape(a.todense(), a.shape)

  c = a.transpose(order)
  npc = npa.transpose(order)
  d = op(c, b)
  npd = op(npc, npb)
  np.testing.assert_allclose(d.todense(), npd)


@pytest.mark.parametrize('op', [np.add, np.subtract])
def test_add_sub_raises(op):
  np.random.seed(10)
  Ds1 = [3, 4, 5, 6]
  Ds2 = [4, 5, 6, 7]

  indices1 = [
      Index(U1Charge.random(dimension=Ds1[n], minval=-5, maxval=5), False)
      for n in range(4)
  ]
  indices2 = [
      Index(U1Charge.random(dimension=Ds2[n], minval=-5, maxval=5), False)
      for n in range(4)
  ]
  a = BlockSparseTensor.randn(indices1)
  b = BlockSparseTensor.randn(indices2)
  with pytest.raises(TypeError):
    op(a, np.array([1, 2, 3]))
  with pytest.raises(ValueError):
    op(a, b)

  Ds3 = [3, 3, 3, 3]
  Ds4 = [9, 9]
  indices3 = [
      Index(U1Charge.random(dimension=Ds3[n], minval=-5, maxval=5), False)
      for n in range(4)
  ]
  indices4 = [
      Index(U1Charge.random(dimension=Ds4[n], minval=-5, maxval=5), False)
      for n in range(2)
  ]
  c = BlockSparseTensor.randn(indices3).reshape([9, 9])
  d = BlockSparseTensor.randn(indices4)
  with pytest.raises(ValueError):
    op(c, d)

  Ds5 = [200, 200]
  Ds6 = [200, 200]
  indices5 = [
      Index(U1Charge.random(dimension=Ds5[n], minval=-5, maxval=5), False)
      for n in range(2)
  ]
  indices6 = [
      Index(U1Charge.random(dimension=Ds6[n], minval=-5, maxval=5), False)
      for n in range(2)
  ]
  e = BlockSparseTensor.randn(indices5)
  f = BlockSparseTensor.randn(indices6)
  with pytest.raises(ValueError):
    op(e, f)


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_mul(dtype, num_charges, chargetype):
  np.random.seed(10)
  indices = [
      Index(get_charge(chargetype, num_charges, 20), False) for _ in range(4)
  ]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = 5 * a
  np.testing.assert_allclose(b.data, a.data * 5)


def test_mul_raises():
  np.random.seed(10)
  indices = [
      Index(U1Charge.random(dimension=10, minval=-5, maxval=5), False)
      for _ in range(4)
  ]
  a = BlockSparseTensor.randn(indices)
  with pytest.raises(TypeError):
    [1, 2] * a


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_rmul(dtype, num_charges, chargetype):
  np.random.seed(10)
  indices = [
      Index(get_charge(chargetype, num_charges, 20), False) for _ in range(4)
  ]

  indices = [
      Index(U1Charge.random(dimension=10, minval=-5, maxval=5), False)
      for _ in range(4)
  ]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = a * 5
  np.testing.assert_allclose(b.data, a.data * 5)


def test_rmul_raises():
  np.random.seed(10)
  indices = [
      Index(U1Charge.random(dimension=10, minval=-5, maxval=5), False)
      for _ in range(4)
  ]
  a = BlockSparseTensor.randn(indices)
  with pytest.raises(TypeError):
    _ = a * np.array([1, 2])


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_truediv(dtype, num_charges, chargetype):
  np.random.seed(10)
  indices = [
      Index(get_charge(chargetype, num_charges, 20), False) for _ in range(4)
  ]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = a / 5
  np.testing.assert_allclose(b.data, a.data / 5)


def test_truediv_raises():
  np.random.seed(10)
  indices = [
      Index(U1Charge.random(dimension=10, minval=-5, maxval=5), False)
      for _ in range(4)
  ]
  a = BlockSparseTensor.randn(indices)
  with pytest.raises(TypeError):
    _ = a / np.array([1, 2])


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_conj(dtype, num_charges, chargetype):
  np.random.seed(10)
  indices = [
      Index(get_charge(chargetype, num_charges, 20), False) for _ in range(4)
  ]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = a.conj()
  np.testing.assert_allclose(b.data, np.conj(a.data))


@pytest.mark.parametrize("rank1", [1, 2])
@pytest.mark.parametrize("rank2", [1, 2])
@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2])
@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
def test_matmul(dtype, num_charges, chargetype, rank1, rank2):
  np.random.seed(10)
  Ds1 = [20] * rank1
  Ds2 = [20] * (rank2 - 1)
  is1 = [
      Index(get_charge(chargetype, num_charges, Ds1[n]), False)
      for n in range(rank1)
  ]
  is2 = [is1[-1].copy().flip_flow()] + [
      Index(get_charge(chargetype, num_charges, Ds2[n]), False)
      for n in range(rank2 - 1)
  ]
  tensor1 = BlockSparseTensor.random(is1, dtype=dtype)
  tensor2 = BlockSparseTensor.random(is2, dtype=dtype)
  result = tensor1 @ tensor2
  assert result.dtype == dtype
  #pylint:disable=line-too-long
  dense_result = tensor1.todense() @ tensor2.todense()  #pytype: disable=unsupported-operands
  np.testing.assert_allclose(dense_result, result.todense())


def test_matmul_raises():
  dtype = np.float64
  num_charges = 1
  np.random.seed(10)
  Ds1 = [100, 200, 20]
  is1 = [
      Index(
          BaseCharge(
              np.random.randint(-5, 5, (Ds1[n], num_charges), dtype=np.int16),
              charge_types=[U1Charge] * num_charges), False) for n in range(3)
  ]
  is2 = [
      is1[1].copy().flip_flow(),
      Index(
          BaseCharge(
              np.random.randint(-5, 5, (150, num_charges), dtype=np.int16),
              charge_types=[U1Charge] * num_charges), False)
  ]
  tensor1 = BlockSparseTensor.random(is1, dtype=dtype)
  tensor2 = BlockSparseTensor.random(is2, dtype=dtype)
  with pytest.raises(ValueError):
    tensor1 @ tensor2
  with pytest.raises(ValueError):
    tensor2 @ tensor1


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_BlockSparseTensor_contiguous(num_charges, chargetype):
  np.random.seed(10)
  Ds = np.array([8, 9, 10, 11])
  order = [2, 0, 1, 3]
  flows = [True, False, True, False]
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), flows[n])
      for n in range(4)
  ]
  arr = BlockSparseTensor.random(indices)
  data1 = np.ascontiguousarray(np.transpose(arr.todense(), order))
  data2 = arr.transpose(order).contiguous().todense()
  np.testing.assert_allclose(data1.strides, data2.strides)
  np.testing.assert_allclose(data1, data2)


def test_BlockSparseTensor_contiguous_1():
  Ds = [10, 11, 12, 13]
  charges = [U1Charge.random(Ds[n], -5, 5) for n in range(4)]
  flows = [True, False, True, False]
  inds = [Index(c, flows[n]) for n, c in enumerate(charges)]
  b = BlockSparseTensor.random(inds, dtype=np.float64)
  order = [0, 3, 2, 1]
  b = b.transpose(order)
  b_ = b.contiguous(inplace=False)
  np.testing.assert_allclose(b.flows, b_.flows)
  for n in range(4):
    assert charge_equal(b._charges[order[n]], b_._charges[n])


def test_BlockSparseTensor_contiguous_2():
  Ds = [10, 11, 12, 13]
  charges = [U1Charge.random(Ds[n], -5, 5) for n in range(4)]
  flows = [True, False, True, False]
  inds = [Index(c, flows[n]) for n, c in enumerate(charges)]
  b = BlockSparseTensor.random(inds, dtype=np.float64)
  order = [0, 3, 2, 1]
  b = b.transpose(order)
  b_ = b.contiguous(inplace=False)
  b.contiguous(inplace=True)
  np.testing.assert_allclose(b.flows, b_.flows)
  for n in range(4):
    assert charge_equal(b._charges[n], b_._charges[n])


def test_BlockSparseTensor_contiguous_3():
  Ds = [10, 11, 12, 13]
  charges = [U1Charge.random(Ds[n], -5, 5) for n in range(4)]
  flows = [True, False, True, False]
  inds = [Index(c, flows[n]) for n, c in enumerate(charges)]
  a = BlockSparseTensor.random(inds, dtype=np.float64)
  order_a = [0, 3, 1, 2]
  a = a.transpose(order_a)
  adense = a.todense()
  order_b = [0, 2, 1, 3]
  b = BlockSparseTensor.random([inds[n] for n in order_b], dtype=np.float64)
  b = b.transpose([0, 3, 2, 1])
  b.contiguous([0, 2, 1, 3], inplace=True)
  bdense = b.todense()
  c = a + b
  cdense = c.todense()
  np.testing.assert_allclose(a.flows, b.flows)
  np.testing.assert_allclose(a.flows, b.flows)
  np.testing.assert_allclose(adense + bdense, cdense)


def test_flat_flows():
  Ds = [10, 11, 12, 13]
  charges = [U1Charge.random(Ds[n], -5, 5) for n in range(4)]
  flows = [True, False, True, False]
  inds = [Index(c, flows[n]) for n, c in enumerate(charges)]
  a = BlockSparseTensor.random(inds, dtype=np.float64)
  order = [0, 3, 1, 2]
  a = a.transpose(order)
  np.testing.assert_allclose(a.flat_flows, [a._flows[o] for o in a.flat_order])


def test_flat_charges():
  Ds = [10, 11, 12, 13]
  charges = [U1Charge.random(Ds[n], -5, 5) for n in range(4)]
  flows = [True, False, True, False]
  inds = [Index(c, flows[n]) for n, c in enumerate(charges)]
  a = BlockSparseTensor.random(inds, dtype=np.float64)
  order = [0, 3, 1, 2]
  a = a.transpose(order)
  for n, o in enumerate(a.flat_order):
    charge_equal(a.flat_charges[n], a._charges[o])


def test_item():
  t1 = BlockSparseTensor(
      data=np.array(1.0),
      charges=[],
      flows=[],
      order=[],
      check_consistency=False)
  assert t1.item() == 1


  Ds = [1, 1]
  charges = [U1Charge.random(Ds[n], 1, 2) for n in range(2)]
  flows = [False, False]
  inds = [Index(c, flows[n]) for n, c in enumerate(charges)]
  t2 = BlockSparseTensor.random(inds, dtype=np.float64)
  assert t2.item() == 0.0

  Ds = [10, 11, 12, 13]
  charges = [U1Charge.random(Ds[n], -5, 5) for n in range(4)]
  flows = [True, False, True, False]
  inds = [Index(c, flows[n]) for n, c in enumerate(charges)]
  t3 = BlockSparseTensor.random(inds, dtype=np.float64)
  with pytest.raises(
      ValueError,
      match="can only convert an array of"
      " size 1 to a Python scalar"):
    t3.item()


@pytest.mark.parametrize('chargetype', ["U1", "Z2"])
@pytest.mark.parametrize('dtype', np_dtypes)
def test_T(chargetype, dtype):
  np.random.seed(10)
  D = 10
  rank = 2
  charges = [get_charge(chargetype, 1, D) for _ in range(rank)]
  flows = np.random.choice([True, False], size=rank, replace=True)
  inds = [Index(c, f) for c, f in zip(charges, flows)]
  T = ChargeArray.random(inds, dtype=dtype)
  TT = T.T
  np.testing.assert_allclose(TT.todense(), T.todense().T)

@pytest.mark.parametrize('chargetype', ["U1", "Z2"])
@pytest.mark.parametrize('dtype', np_dtypes)
def test_herm(chargetype, dtype):
  np.random.seed(10)
  D = 10
  rank = 2
  charges = [get_charge(chargetype, 1, D) for _ in range(rank)]
  flows = np.random.choice([True, False], size=rank, replace=True)
  inds = [Index(c, f) for c, f in zip(charges, flows)]
  T = ChargeArray.random(inds, dtype=dtype)
  TH = T.H
  np.testing.assert_allclose(TH.todense(), T.todense().T.conj())


@pytest.mark.parametrize('chargetype', ["U1", "Z2"])
@pytest.mark.parametrize('dtype', np_dtypes)
def test_herm_raises(chargetype, dtype):
  np.random.seed(10)
  D = 10
  rank = 3
  charges = [get_charge(chargetype, 1, D) for _ in range(rank)]
  flows = np.random.choice([True, False], size=rank, replace=True)
  inds = [Index(c, f) for c, f in zip(charges, flows)]
  T = ChargeArray.random(inds, dtype=dtype)
  with pytest.raises(ValueError, match="hermitian"):
    T.H

@pytest.mark.parametrize('chargetype', ["U1", "Z2"])
@pytest.mark.parametrize('dtype', np_dtypes)
def test_neg(chargetype, dtype):
  np.random.seed(10)
  D = 10
  rank = 2
  charges = [get_charge(chargetype, 1, D) for _ in range(rank)]
  flows = np.random.choice([True, False], size=rank, replace=True)
  inds = [Index(c, f) for c, f in zip(charges, flows)]
  T = BlockSparseTensor.random(inds, dtype=dtype)
  T2 = -T
  np.testing.assert_allclose(T.data, -T2.data)

def test_ChargeArray_arithmetic_raises():
  np.random.seed(10)
  dtype = np.float64
  D = 10
  rank = 3
  charges = [U1Charge.random(D, -2, 2) for _ in range(rank)]
  flows = np.random.choice([True, False], size=rank, replace=True)
  inds = [Index(c, f) for c, f in zip(charges, flows)]
  T = ChargeArray.random(inds, dtype=dtype)
  with pytest.raises(NotImplementedError):
    T - T
  with pytest.raises(NotImplementedError):
    T + T
  with pytest.raises(NotImplementedError):
    -T
  with pytest.raises(NotImplementedError):
    T * 5
  with pytest.raises(NotImplementedError):
    5 * T
  with pytest.raises(NotImplementedError):
    T / 5


def test_repr():
  np.random.seed(10)
  dtype = np.float64
  D = 10
  rank = 3
  charges = [U1Charge.random(D, -2, 2) for _ in range(rank)]
  flows = np.random.choice([True, False], size=rank, replace=True)
  inds = [Index(c, f) for c, f in zip(charges, flows)]
  T = ChargeArray.random(inds, dtype=dtype)
  actual = T.__repr__()
  expected = "ChargeArray\n   shape: (10, 10, 10)\n  " +\
    " charge types: ['U1Charge']\n   dtype: " +\
    repr(T.dtype.name) + "\n   flat flows: " + \
    repr(list(flows)) + "\n   order: " + repr(T._order)
  assert actual == expected

  res = tensordot(T, T.conj(), ([0, 1, 2], [0, 1, 2]))
  actual = res.__repr__()
  expected = "BlockSparseTensor\n   shape: ()\n  " +\
    " charge types: no charge types (scalar)\n   dtype: " +\
    repr(res.dtype.name) + "\n   flat flows: " + \
    repr(list(res.flat_flows)) + "\n   order: " + repr(res._order)
  assert actual == expected

@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_size(chargetype, num_charges):
  np.random.seed(10)
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), flows[n])
      for n in range(4)
  ]
  arr = BlockSparseTensor.random(indices)
  assert arr.size == np.prod(Ds)

@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_compare_shapes(chargetype, num_charges):
  np.random.seed(10)
  Ds1 = np.array([8, 9, 10, 11])
  flows1 = [True, False, True, False]
  indices1 = [
      Index(get_charge(chargetype, num_charges, Ds1[n]), flows1[n])
      for n in range(4)
  ]
  indices2 = [
      Index(get_charge(chargetype, num_charges, Ds1[n]), flows1[n])
      for n in range(4)
  ]

  Ds3 = np.array([8, 12, 13])
  flows3 = [False, True, False]
  indices3 = [
      Index(get_charge(chargetype, num_charges, Ds3[n]), flows3[n])
      for n in range(3)
  ]

  Ds4 = np.array([2, 4, 9, 10, 11])
  flows4 = [False, True, False, True, False]
  indices4 = [
      Index(get_charge(chargetype, num_charges, Ds4[n]), flows4[n])
      for n in range(len(Ds4))
  ]
  arr1 = BlockSparseTensor.random(indices1)
  arr2 = BlockSparseTensor.random(indices2)
  arr3 = BlockSparseTensor.random(indices3)
  arr4 = BlockSparseTensor.random(indices4).reshape(Ds1)

  assert compare_shapes(arr1, arr1)
  assert not compare_shapes(arr1, arr2)
  assert not compare_shapes(arr1, arr4)
  assert not compare_shapes(arr1, arr3)
  assert not compare_shapes(arr2, arr3)
