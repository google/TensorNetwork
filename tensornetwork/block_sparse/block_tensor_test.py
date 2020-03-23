import numpy as np
import pytest
import itertools
# pylint: disable=line-too-long
from tensornetwork.block_sparse.charge import U1Charge, fuse_charges, charge_equal, fuse_ndarrays, fuse_ndarray_charges, BaseCharge
from tensornetwork.block_sparse.index import Index
from tensornetwork import ncon
# pylint: disable=line-too-long

from tensornetwork.block_sparse.block_tensor import flatten, get_flat_meta_data, fuse_stride_arrays, compute_sparse_lookup, _find_best_partition, compute_fused_charge_degeneracies, compute_unique_fused_charges, compute_num_nonzero, reduce_charges, _find_diagonal_sparse_blocks, _get_strides, _find_transposed_diagonal_sparse_blocks, ChargeArray, BlockSparseTensor, norm, diag, reshape, transpose, conj, outerproduct

np_dtypes = [np.float64, np.complex128]
np_tensordot_dtypes = [np.float64, np.complex128]


def test_flatten():
  listoflist = [[1, 2], [3, 4], [5]]
  flat = flatten(listoflist)
  np.testing.assert_allclose(flat, [1, 2, 3, 4, 5])


def test_flat_meta_data():
  i1 = Index([U1Charge.random(-2, 2, 20),
              U1Charge.random(-2, 2, 20)],
             flow=[True, False])

  i2 = Index([U1Charge.random(-2, 2, 20),
              U1Charge.random(-2, 2, 20)],
             flow=[False, True])
  expected_charges = [
      i1._charges[0], i1._charges[1], i2._charges[0], i2._charges[1]
  ]
  expected_flows = [True, False, False, True]
  charges, flows = get_flat_meta_data([i1, i2])
  np.testing.assert_allclose(flows, expected_flows)
  for n, c in enumerate(charges):
    assert charge_equal(c, expected_charges[n])


def test_fuse_stride_arrays():
  dims = np.asarray([2, 3, 4, 5])
  strides = np.asarray([120, 60, 20, 5, 1])
  actual = fuse_stride_arrays(dims, strides)
  expected = fuse_ndarrays([
      np.arange(0, strides[n] * dims[n], strides[n], dtype=np.uint32)
      for n in range(len(dims))
  ])
  np.testing.assert_allclose(actual, expected)


def test_compute_sparse_lookup():
  q1 = np.array([-2, 0, -5, 7])
  q2 = np.array([-3, 1, -2, 6, 2, -2])
  expected_unique = np.array(
      [-11, -8, -7, -6, -4, -3, -2, -1, 0, 1, 2, 3, 5, 6, 9, 10])

  expected_labels_to_unique = np.array([7, 8, 9])
  expected_lookup = np.array([9, 8, 8, 7, 9])

  charges = [U1Charge(q1), U1Charge(q2)]
  targets = U1Charge(np.array([-1, 0, 1]))
  flows = [False, True]
  lookup, unique, labels = compute_sparse_lookup(charges, flows, targets)
  np.testing.assert_allclose(lookup, expected_lookup)
  np.testing.assert_allclose(expected_unique, np.squeeze(unique.charges))
  np.testing.assert_allclose(labels, expected_labels_to_unique)


def test_find_best_partition():
  d = [5, 4, 5, 2, 6, 8]
  p = _find_best_partition(d)
  assert p == 3


def test_compute_fused_charge_degeneracies():
  np.random.seed(10)
  qs = [np.random.randint(-3, 3, 100) for _ in range(3)]
  charges = [U1Charge(q) for q in qs]
  flows = [False, True, False]
  np_flows = [1, -1, 1]
  unique, degens = compute_fused_charge_degeneracies(charges, flows)
  fused = fuse_ndarrays([qs[n] * np_flows[n] for n in range(3)])
  exp_unique, exp_degens = np.unique(fused, return_counts=True)
  np.testing.assert_allclose(np.squeeze(unique.charges), exp_unique)
  np.testing.assert_allclose(degens, exp_degens)


def test_compute_unique_fused_charges():
  np.random.seed(10)
  qs = [np.random.randint(-3, 3, 100) for _ in range(3)]
  charges = [U1Charge(q) for q in qs]
  flows = [False, True, False]
  np_flows = [1, -1, 1]
  unique = compute_unique_fused_charges(charges, flows)
  fused = fuse_ndarrays([qs[n] * np_flows[n] for n in range(3)])
  exp_unique = np.unique(fused)
  np.testing.assert_allclose(np.squeeze(unique.charges), exp_unique)


def test_compute_num_nonzero():
  np.random.seed(10)
  qs = [np.random.randint(-3, 3, 100) for _ in range(3)]
  charges = [U1Charge(q) for q in qs]
  flows = [False, True, False]
  np_flows = [1, -1, 1]
  fused = fuse_ndarrays([qs[n] * np_flows[n] for n in range(3)])
  nz1 = compute_num_nonzero(charges, flows)
  nz2 = len(np.nonzero(fused == 0)[0])
  assert nz1 == nz2


def test_reduce_charges():
  left_charges = np.asarray([-2, 0, 1, 0, 0]).astype(np.int16)
  right_charges = np.asarray([-1, 0, 2, 1]).astype(np.int16)
  target_charge = np.zeros((1, 1), dtype=np.int16)
  fused_charges = fuse_ndarrays([left_charges, right_charges])
  dense_positions = reduce_charges(
      [U1Charge(left_charges), U1Charge(right_charges)], [False, False],
      target_charge,
      return_locations=True)

  np.testing.assert_allclose(dense_positions[0].charges, 0)

  np.testing.assert_allclose(
      dense_positions[1],
      np.nonzero(fused_charges == target_charge[0, 0])[0])


def test_reduce_charges_non_trivial():
  np.random.seed(10)
  left_charges = np.random.randint(-5, 5, 200, dtype=np.int16)
  right_charges = np.random.randint(-5, 5, 200, dtype=np.int16)

  target_charge = np.array([[-2, 0, 3]]).astype(np.int16)
  fused_charges = fuse_ndarrays([left_charges, right_charges])
  dense_positions = reduce_charges(
      [U1Charge(left_charges), U1Charge(right_charges)], [False, False],
      target_charge,
      return_locations=True)
  assert np.all(
      np.isin(
          np.squeeze(dense_positions[0].charges), np.squeeze(target_charge)))
  mask = np.isin(fused_charges, np.squeeze(target_charge))
  np.testing.assert_allclose(dense_positions[1], np.nonzero(mask)[0])


def test_reduce_charges_non_trivial_2():
  np.random.seed(10)
  left_charges1 = np.random.randint(-5, 5, 200, dtype=np.int16)
  left_charges2 = np.random.randint(-5, 5, 200, dtype=np.int16)
  right_charges1 = np.random.randint(-5, 5, 200, dtype=np.int16)
  right_charges2 = np.random.randint(-5, 5, 200, dtype=np.int16)

  target_charge = np.array([[-2, 0, 3], [-1, 1, 0]]).astype(np.int16)
  fused_charges1 = fuse_ndarrays([left_charges1, right_charges1])
  fused_charges2 = fuse_ndarrays([left_charges2, right_charges2])

  dense_positions = reduce_charges([
      U1Charge(left_charges1) @ U1Charge(left_charges2),
      U1Charge(right_charges1) @ U1Charge(right_charges2)
  ], [False, False],
                                   target_charge,
                                   return_locations=True)
  masks = []
  assert np.all(dense_positions[0].isin(target_charge))
  #pylint: disable=unsubscriptable-object
  for n in range(target_charge.shape[1]):
    mask1 = np.isin(fused_charges1, np.squeeze(target_charge[0, n]))
    mask2 = np.isin(fused_charges2, np.squeeze(target_charge[1, n]))
    masks.append(np.logical_and(mask1, mask2))
  #pylint: disable=no-member
  np.testing.assert_allclose(
      np.nonzero(np.logical_or.reduce(masks))[0], dense_positions[1])


@pytest.mark.parametrize('num_legs', [2, 3, 4])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_find_diagonal_sparse_blocks(num_legs, num_charges):
  np.random.seed(10)
  np_charges = [
      np.random.randint(-5, 5, (num_charges, 60), dtype=np.int16)
      for _ in range(num_legs)
  ]
  fused = np.stack([
      fuse_ndarrays([np_charges[n][c, :]
                     for n in range(num_legs)])
      for c in range(num_charges)
  ],
                   axis=0)

  left_charges = np.stack([
      fuse_ndarrays([np_charges[n][c, :]
                     for n in range(num_legs // 2)])
      for c in range(num_charges)
  ],
                          axis=0)
  right_charges = np.stack([
      fuse_ndarrays(
          [np_charges[n][c, :]
           for n in range(num_legs // 2, num_legs)])
      for c in range(num_charges)
  ],
                           axis=0)
  #pylint: disable=no-member
  nz = np.nonzero(
      np.logical_and.reduce(fused.T == np.zeros((1, num_charges)), axis=1))[0]
  linear_locs = np.arange(len(nz))
  # pylint: disable=no-member
  left_inds, _ = np.divmod(nz, right_charges.shape[1])
  left = left_charges[:, left_inds]
  unique_left = np.unique(left, axis=1)
  blocks = []
  for n in range(unique_left.shape[1]):
    ul = unique_left[:, n][None, :]
    #pylint: disable=no-member
    blocks.append(linear_locs[np.nonzero(
        np.logical_and.reduce(left.T == ul, axis=1))[0]])

  charges = [
      BaseCharge(left_charges, charge_types=[U1Charge] * num_charges),
      BaseCharge(right_charges, charge_types=[U1Charge] * num_charges)
  ]
  bs, cs, ss = _find_diagonal_sparse_blocks(charges, [False, False], 1)
  np.testing.assert_allclose(cs.charges, unique_left)
  for b1, b2 in zip(blocks, bs):
    assert np.all(b1 == b2)

  assert np.sum(np.prod(ss, axis=0)) == np.sum([len(b) for b in bs])
  np.testing.assert_allclose(unique_left, cs.charges)


orders = []
bonddims = []
for dim, nl in zip([60, 30, 20], [2, 3, 4]):
  o = list(itertools.permutations(np.arange(nl)))
  orders.extend(o)
  bonddims.extend([dim] * len(o))


@pytest.mark.parametrize('order,D', zip(orders, bonddims))
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_find_transposed_diagonal_sparse_blocks(num_charges, order, D):
  order = list(order)
  num_legs = len(order)
  np.random.seed(10)
  np_charges = [
      np.random.randint(-5, 5, (num_charges, D), dtype=np.int16)
      for _ in range(num_legs)
  ]
  tr_charge_list = []
  charge_list = []
  for c in range(num_charges):
    tr_charge_list.append(
        fuse_ndarrays([np_charges[order[n]][c, :] for n in range(num_legs)]))
    charge_list.append(
        fuse_ndarrays([np_charges[n][c, :] for n in range(num_legs)]))

  tr_fused = np.stack(tr_charge_list, axis=0)
  fused = np.stack(charge_list, axis=0)

  dims = [c.shape[1] for c in np_charges]
  strides = _get_strides(dims)
  transposed_linear_positions = fuse_stride_arrays(dims,
                                                   [strides[o] for o in order])
  left_charges = np.stack([
      fuse_ndarrays([np_charges[order[n]][c, :]
                     for n in range(num_legs // 2)])
      for c in range(num_charges)
  ],
                          axis=0)
  right_charges = np.stack([
      fuse_ndarrays(
          [np_charges[order[n]][c, :]
           for n in range(num_legs // 2, num_legs)])
      for c in range(num_charges)
  ],
                           axis=0)
  #pylint: disable=no-member
  mask = np.logical_and.reduce(fused.T == np.zeros((1, num_charges)), axis=1)
  nz = np.nonzero(mask)[0]
  dense_to_sparse = np.empty(len(mask), dtype=np.int64)
  dense_to_sparse[mask] = np.arange(len(nz))

  tr_mask = np.logical_and.reduce(
      tr_fused.T == np.zeros((1, num_charges)), axis=1)
  tr_nz = np.nonzero(tr_mask)[0]
  tr_linear_locs = transposed_linear_positions[tr_nz]
  # pylint: disable=no-member
  left_inds, _ = np.divmod(tr_nz, right_charges.shape[1])
  left = left_charges[:, left_inds]
  unique_left = np.unique(left, axis=1)
  blocks = []
  for n in range(unique_left.shape[1]):
    ul = unique_left[:, n][None, :]
    #pylint: disable=no-member
    blocks.append(dense_to_sparse[tr_linear_locs[np.nonzero(
        np.logical_and.reduce(left.T == ul, axis=1))[0]]])

  charges = [
      BaseCharge(c, charge_types=[U1Charge] * num_charges) for c in np_charges
  ]
  flows = [False] * num_legs
  bs, cs, ss = _find_transposed_diagonal_sparse_blocks(
      charges, flows, tr_partition=num_legs // 2, order=order)
  np.testing.assert_allclose(cs.charges, unique_left)
  for b1, b2 in zip(blocks, bs):
    assert np.all(b1 == b2)

  assert np.sum(np.prod(ss, axis=0)) == np.sum([len(b) for b in bs])
  np.testing.assert_allclose(unique_left, cs.charges)


def test_ChargeArray_init():
  np.random.seed(10)
  D = 10
  rank = 4
  charges = [U1Charge.random(-5, 5, D) for _ in range(rank)]
  data = np.random.uniform(0, 1, size=D**rank)
  flows = np.random.choice([True, False], size=rank, replace=True)
  order = [[n] for n in range(rank)]
  arr = ChargeArray(data, charges, flows, order=order)
  np.testing.assert_allclose(data, arr.data)
  for c1, c2 in zip(charges, arr.charges):
    assert charge_equal(c1, c2[0])
  for c1, c2 in zip(charges, arr._charges):
    assert charge_equal(c1, c2)


@pytest.mark.parametrize('dtype', np_dtypes)
def test_ChargeArray_generic(dtype):
  Ds = [8, 9, 10, 11]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), False) for n in range(4)]
  arr = ChargeArray.random(indices, dtype=dtype)
  assert arr.ndim == 4
  assert arr.dtype == dtype
  np.testing.assert_allclose(arr.shape, Ds)
  np.testing.assert_allclose(arr.flat_flows, [False, False, False, False])
  for n in range(4):
    assert charge_equal(indices[n]._charges[0], arr.flat_charges[n])
    assert arr.sparse_shape[n] == indices[n]


@pytest.mark.parametrize('dtype', np_dtypes)
def test_ChargeArray_todense(dtype):
  Ds = [8, 9, 10, 11]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), False) for n in range(4)]
  arr = ChargeArray.random(indices, dtype=dtype)
  np.testing.assert_allclose(arr.todense(), np.reshape(arr.data, Ds))


@pytest.mark.parametrize('dtype', np_dtypes)
def test_ChargeArray_reshape(dtype):
  Ds = [8, 9, 10, 11]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), False) for n in range(4)]
  arr = ChargeArray.random(indices, dtype=dtype)
  arr2 = arr.reshape([72, 110])
  for n in range(2):
    for m in range(2):
      assert charge_equal(arr2.charges[n][m], indices[n * 2 + m].charges)
  np.testing.assert_allclose(arr2.shape, [72, 110])
  np.testing.assert_allclose(arr2._order, [[0, 1], [2, 3]])
  np.testing.assert_allclose(arr2.flows, [[False, False], [False, False]])
  assert arr2.ndim == 2
  arr3 = arr.reshape(Ds)
  for n in range(4):
    assert charge_equal(arr3.charges[n][0], indices[n].charges)

  np.testing.assert_allclose(arr3.shape, Ds)
  np.testing.assert_allclose(arr3._order, [[0], [1], [2], [3]])
  np.testing.assert_allclose(arr3.flows, [[False], [False], [False], [False]])
  assert arr3.ndim == 4


def test_ChargeArray_reshape_raises():
  Ds = [8, 9, 10, 11]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), False) for n in range(4)]
  arr = ChargeArray.random(indices)
  with pytest.raises(ValueError):
    arr.reshape([64, 65])

  arr2 = arr.reshape([72, 110])
  with pytest.raises(ValueError):
    arr2.reshape([9, 8, 10, 11])


def test_transpose():
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(4)]
  arr = ChargeArray.random(indices)
  order = [2, 1, 0, 3]
  arr2 = arr.transpose(order)
  np.testing.assert_allclose(Ds[order], arr2.shape)
  np.testing.assert_allclose(arr2._order, [[2], [1], [0], [3]])
  np.testing.assert_allclose(arr2.flows, [[True], [False], [True], [False]])


def test_transpose_reshape():
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(4)]
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


def test_transpose_data():
  Ds = np.array([8, 9, 10, 11])
  order = [2, 0, 1, 3]
  flows = [True, False, True, False]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(4)]
  arr = ChargeArray.random(indices)
  data = np.ascontiguousarray(np.transpose(np.reshape(arr.data, Ds), order))
  arr2 = arr.transpose(order).transpose_data()
  data3 = np.reshape(arr2.data, Ds[order])
  np.testing.assert_allclose(data, data3)
  np.testing.assert_allclose(arr2.shape, Ds[order])
  np.testing.assert_allclose(arr2._order, [[0], [1], [2], [3]])
  np.testing.assert_allclose(arr2.flows, [[True], [True], [False], [False]])


def test_transpose_reshape_transpose_data():
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(4)]
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
  charges = [U1Charge.random(-5, 5, D) for _ in range(rank)]
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


@pytest.mark.parametrize('dtype', np_dtypes)
def test_BlockSparseTensor_random(dtype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(rank)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)

  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('dtype', np_dtypes)
def test_BlockSparseTensor_randn(dtype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(rank)
  ]
  arr = BlockSparseTensor.randn(indices, dtype=dtype)

  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('dtype', np_dtypes)
def test_BlockSparseTensor_ones(dtype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(rank)
  ]
  arr = BlockSparseTensor.ones(indices, dtype=dtype)
  np.testing.assert_allclose(arr.data, 1)
  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('dtype', np_dtypes)
def test_BlockSparseTensor_zeros(dtype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(rank)
  ]
  arr = BlockSparseTensor.zeros(indices, dtype=dtype)
  np.testing.assert_allclose(arr.data, 0)
  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


def test_copy():
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(rank)
  ]
  arr = BlockSparseTensor.randn(indices)
  copy = arr.copy()
  assert arr.data is not copy.data
  for n in range(len(arr._charges)):
    assert arr._charges[n] is not copy._charges[n]
  assert arr._flows is not copy._flows
  assert arr._order is not copy._order


def test_todense():
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  charges = [U1Charge.random(-5, 5, Ds[n]) for n in range(rank)]
  fused = fuse_charges(charges, flows)
  mask = fused == np.zeros((1, 1))
  inds = np.nonzero(mask)[0]
  inds2 = np.nonzero(np.logical_not(mask))[0]
  indices = [Index(charges[n], flows[n]) for n in range(rank)]
  arr = BlockSparseTensor.randn(indices)
  dense = np.array(arr.todense().flat)
  np.testing.assert_allclose(dense[inds], arr.data)
  np.testing.assert_allclose(dense[inds2], 0)


@pytest.mark.parametrize('op', [np.add, np.subtract])
@pytest.mark.parametrize('dtype', np_dtypes)
def test_add_sub(op, dtype):
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
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


@pytest.mark.parametrize('dtype', np_dtypes)
def test_mul(dtype):
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = 5 * a
  np.testing.assert_allclose(b.data, a.data * 5)


@pytest.mark.parametrize('dtype', np_dtypes)
def test_rmul(dtype):
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = a * 5
  np.testing.assert_allclose(b.data, a.data * 5)


@pytest.mark.parametrize('dtype', np_dtypes)
def test_truediv(dtype):
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = a / 5
  np.testing.assert_allclose(b.data, a.data / 5)


@pytest.mark.parametrize('dtype', np_dtypes)
def test_conj(dtype):
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = a.conj()
  np.testing.assert_allclose(b.data, np.conj(a.data))


def test_BlockSparseTensor_transpose_data():
  np.random.seed(10)
  Ds = np.array([8, 9, 10, 11])
  order = [2, 0, 1, 3]
  flows = [True, False, True, False]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(4)]
  arr = BlockSparseTensor.random(indices)
  data1 = np.ascontiguousarray(np.transpose(arr.todense(), order))
  data2 = arr.transpose(order).transpose_data().todense()
  np.testing.assert_allclose(data1.strides, data2.strides)
  np.testing.assert_allclose(data1, data2)


@pytest.mark.parametrize('dtype', np_dtypes)
def test_norm(dtype):
  np.random.seed(10)
  Ds = np.asarray([8, 9, 10, 11])
  rank = Ds.shape[0]
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(4)]
  arr = BlockSparseTensor.random(indices, dtype=dtype)
  dense_norm = np.linalg.norm(arr.todense())
  np.testing.assert_allclose(norm(arr), dense_norm)


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_get_diag(dtype, num_charges):
  np.random.seed(10)
  Ds = [100, 200]
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-2, 3, (num_charges, Ds[n])),
              charge_types=[U1Charge] * num_charges), False) for n in range(2)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)
  fused = fuse_charges(arr.flat_charges, arr.flat_flows)
  inds = np.nonzero(fused == np.zeros((1, 1), dtype=np.int16))[0]
  # pylint: disable=no-member
  left, _ = np.divmod(inds, 200)
  unique = np.unique(indices[0]._charges[0].charges[:, left], axis=1)
  diagonal = diag(arr)
  sparse_blocks, _, block_shapes = _find_diagonal_sparse_blocks(
      arr.flat_charges, arr.flat_flows, 1)
  data = np.concatenate([
      np.diag(np.reshape(arr.data[sparse_blocks[n]], block_shapes[:, n]))
      for n in range(len(sparse_blocks))
  ])
  np.testing.assert_allclose(data, diagonal.data)
  np.testing.assert_allclose(unique, diagonal.flat_charges[0].unique_charges)


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_create_diag(dtype, num_charges):
  np.random.seed(10)
  D = 200
  index = Index(
      BaseCharge(
          np.random.randint(-2, 3, (num_charges, D)),
          charge_types=[U1Charge] * num_charges), False)

  arr = ChargeArray.random([index], dtype=dtype)
  diagarr = diag(arr)
  dense = np.ravel(diagarr.todense())
  np.testing.assert_allclose(
      np.sort(dense[dense != 0.0]), np.sort(diagarr.data[diagarr.data != 0.0]))

  sparse_blocks, charges, block_shapes = _find_diagonal_sparse_blocks(
      diagarr.flat_charges, diagarr.flat_flows, 1)
  #in range(index._charges[0].unique_charges.shape[1]):
  for n, block in enumerate(sparse_blocks):
    shape = block_shapes[:, n]
    block_diag = np.diag(np.reshape(diagarr.data[block], shape))
    np.testing.assert_allclose(
        arr.data[np.squeeze(index._charges[0] == charges[n])], block_diag)


def test_diag_raises():
  np.random.seed(10)
  Ds = [8, 9, 10]
  rank = len(Ds)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-2, 3, (1, Ds[n])), charge_types=[U1Charge]),
          False) for n in range(rank)
  ]
  arr = BlockSparseTensor.random(indices)
  chargearr = ChargeArray.random([indices[0], indices[1]])
  with pytest.raises(ValueError):
    diag(arr)
  with pytest.raises(ValueError):
    diag(chargearr)


@pytest.mark.parametrize('dtype', np_dtypes)
def test_tn_reshape(dtype):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), False) for n in range(4)]
  arr = BlockSparseTensor.random(indices, dtype=dtype)
  arr2 = reshape(arr, [72, 110])
  for n in range(2):
    for m in range(2):
      assert charge_equal(arr2.charges[n][m], indices[n * 2 + m].charges)
  np.testing.assert_allclose(arr2.shape, [72, 110])
  np.testing.assert_allclose(arr2._order, [[0, 1], [2, 3]])
  np.testing.assert_allclose(arr2.flows, [[False, False], [False, False]])
  assert arr2.ndim == 2
  arr3 = reshape(arr, Ds)
  for n in range(4):
    assert charge_equal(arr3.charges[n][0], indices[n].charges)

  np.testing.assert_allclose(arr3.shape, Ds)
  np.testing.assert_allclose(arr3._order, [[0], [1], [2], [3]])
  np.testing.assert_allclose(arr3.flows, [[False], [False], [False], [False]])
  assert arr3.ndim == 4


def test_tn_transpose():
  np.random.seed(10)
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(4)]
  arr = BlockSparseTensor.random(indices)
  order = [2, 1, 0, 3]
  arr2 = transpose(arr, order)
  np.testing.assert_allclose(Ds[order], arr2.shape)
  np.testing.assert_allclose(arr2._order, [[2], [1], [0], [3]])
  np.testing.assert_allclose(arr2.flows, [[True], [False], [True], [False]])


def test_tn_transpose_reshape():
  np.random.seed(10)
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(4)]
  arr = BlockSparseTensor.random(indices)
  arr2 = transpose(arr, [2, 0, 1, 3])
  arr3 = reshape(arr2, [80, 99])
  np.testing.assert_allclose(arr3.shape, [80, 99])
  np.testing.assert_allclose(arr3._order, [[2, 0], [1, 3]])
  np.testing.assert_allclose(arr3.flows, [[True, True], [False, False]])

  arr4 = transpose(arr3, [1, 0])
  np.testing.assert_allclose(arr4.shape, [99, 80])
  np.testing.assert_allclose(arr4._order, [[1, 3], [2, 0]])
  np.testing.assert_allclose(arr4.flows, [[False, False], [True, True]])

  arr5 = reshape(arr4, [9, 11, 10, 8])
  np.testing.assert_allclose(arr5.shape, [9, 11, 10, 8])
  np.testing.assert_allclose(arr5._order, [[1], [3], [2], [0]])
  np.testing.assert_allclose(arr5.flows, [[False], [False], [True], [True]])


@pytest.mark.parametrize('dtype', np_dtypes)
def test_tn_conj(dtype):
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = conj(a)
  np.testing.assert_allclose(b.data, np.conj(a.data))


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_legs', [1, 2, 3, 4])
def test_outerproduct(dtype, num_legs):
  Ds1 = np.arange(2, 2 + num_legs)
  Ds2 = np.arange(2 + num_legs, 2 + 2 * num_legs)
  is1 = [Index(U1Charge.random(-5, 5, Ds1[n]), False) for n in range(num_legs)]
  is2 = [Index(U1Charge.random(-5, 5, Ds2[n]), False) for n in range(num_legs)]
  a = BlockSparseTensor.random(is1, dtype=dtype)
  b = BlockSparseTensor.random(is2, dtype=dtype)
  abdense = ncon([a.todense(), b.todense()], [
      -np.arange(1, num_legs + 1, dtype=np.int16),
      -num_legs - np.arange(1, num_legs + 1, dtype=np.int16)
  ])
  ab = outerproduct(a, b)
  np.testing.assert_allclose(ab.todense(), abdense)
