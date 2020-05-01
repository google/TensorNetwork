import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_sparse.charge import U1Charge, fuse_charges, charge_equal, fuse_ndarrays, fuse_ndarray_charges, BaseCharge, Z2Charge
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparsetensor import ChargeArray, BlockSparseTensor

np_dtypes = [np.float64, np.complex128]
np_tensordot_dtypes = [np.float64, np.complex128]


def get_charge(chargetype, num_charges, D):
  if chargetype == "U1":
    return BaseCharge(
        np.random.randint(-5, 6, (num_charges, D)),
        charge_types=[U1Charge] * num_charges)
  if chargetype == "Z2":
    return BaseCharge(
        np.random.randint(0, 2, (num_charges, D)),
        charge_types=[Z2Charge] * num_charges)
  if chargetype == "mixed":
    n1 = num_charges // 2 if num_charges > 1 else 1
    c = BaseCharge(
        np.random.randint(-5, 6, (n1, D)), charge_types=[U1Charge] * n1)

    if num_charges > 1:
      n2 = num_charges - n1
      c = c @ BaseCharge(
          np.random.randint(0, 2, (n2, D)), charge_types=[Z2Charge] * n2)

    return c
  return None


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
    ChargeArray(data, charges, flows, order=order)


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


def test_ChargeArray_reshape_raises():
  Ds = [8, 9, 10, 11]
  indices = [
      Index(U1Charge.random(dimension=Ds[n], minval=-5, maxval=5), False)
      for n in range(4)
  ]
  arr = ChargeArray.random(indices)
  with pytest.raises(ValueError):
    arr.reshape([64, 65])

  arr2 = arr.reshape([72, 110])
  with pytest.raises(ValueError):
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
def test_ChargeArray_transpose_data(num_charges, chargetype):
  Ds = np.array([8, 9, 10, 11])
  order = [2, 0, 1, 3]
  flows = [True, False, True, False]
  indices = [
      Index(get_charge(chargetype, num_charges, Ds[n]), flows[n])
      for n in range(4)
  ]
  arr = ChargeArray.random(indices)
  data = np.ascontiguousarray(np.transpose(np.reshape(arr.data, Ds), order))
  arr2 = arr.transpose(order).transpose_data()
  data3 = np.reshape(arr2.data, Ds[order])
  np.testing.assert_allclose(data, data3)
  np.testing.assert_allclose(arr2.shape, Ds[order])
  np.testing.assert_allclose(arr2._order, [[0], [1], [2], [3]])
  np.testing.assert_allclose(arr2.flows, [[True], [True], [False], [False]])


@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_ChargeArray_transpose_reshape_transpose_data(num_charges, chargetype):
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
  np.testing.assert_allclose(arr3.transpose_data().data,
                             np.ascontiguousarray(nparr3).flat)
  np.testing.assert_allclose(arr4.transpose_data().data,
                             np.ascontiguousarray(nparr4).flat)
  np.testing.assert_allclose(arr5.transpose_data().data,
                             np.ascontiguousarray(nparr5).flat)


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
  mask = fused == np.zeros((num_charges, 1))
  inds = np.nonzero(mask)[0]
  inds2 = np.nonzero(np.logical_not(mask))[0]
  indices = [Index(charges[n], flows[n]) for n in range(rank)]
  arr = BlockSparseTensor.randn(indices)
  dense = np.array(arr.todense().flat)
  np.testing.assert_allclose(dense[inds], arr.data)
  np.testing.assert_allclose(dense[inds2], 0)


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


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
@pytest.mark.parametrize('chargetype', ["U1", "Z2", "mixed"])
def test_matmul(dtype, num_charges, chargetype):
  np.random.seed(10)
  Ds1 = [100, 200]
  is1 = [
      Index(get_charge(chargetype, num_charges, Ds1[n]), False)
      for n in range(2)
  ]
  is2 = [
      is1[1].copy().flip_flow(),
      Index(get_charge(chargetype, num_charges, 150), False)
  ]
  tensor1 = BlockSparseTensor.random(is1, dtype=dtype)
  tensor2 = BlockSparseTensor.random(is2, dtype=dtype)
  result = tensor1 @ tensor2
  assert result.dtype == dtype

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
              np.random.randint(-5, 5, (num_charges, Ds1[n]), dtype=np.int16),
              charge_types=[U1Charge] * num_charges), False) for n in range(3)
  ]
  is2 = [
      is1[1].copy().flip_flow(),
      Index(
          BaseCharge(
              np.random.randint(-5, 5, (num_charges, 150), dtype=np.int16),
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
def test_BlockSparseTensor_transpose_data(num_charges, chargetype):
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
  data2 = arr.transpose(order).transpose_data().todense()
  np.testing.assert_allclose(data1.strides, data2.strides)
  np.testing.assert_allclose(data1, data2)
