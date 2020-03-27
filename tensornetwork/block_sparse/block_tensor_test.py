import numpy as np
import pytest
import itertools
# pylint: disable=line-too-long
from tensornetwork.block_sparse.charge import U1Charge, fuse_charges, charge_equal, fuse_ndarrays, fuse_ndarray_charges, BaseCharge
from tensornetwork.block_sparse.index import Index
from tensornetwork import ncon
# pylint: disable=line-too-long
from tensornetwork.block_sparse.block_tensor import flatten, get_flat_meta_data, fuse_stride_arrays, compute_sparse_lookup, _find_best_partition, compute_fused_charge_degeneracies, compute_unique_fused_charges, compute_num_nonzero, reduce_charges, _find_diagonal_sparse_blocks, _get_strides, _find_transposed_diagonal_sparse_blocks, ChargeArray, BlockSparseTensor, norm, diag, reshape, transpose, conj, outerproduct, tensordot, svd, qr, eigh, eig, inv, sqrt, trace, eye, pinv, zeros, ones, randn, random

np_dtypes = [np.float64, np.complex128]
np_tensordot_dtypes = [np.float64, np.complex128]


def fuse_many_ndarray_charges(charges, charge_types):
  res = fuse_ndarray_charges(charges[0], charges[1], charge_types)
  for n in range(2, len(charges)):
    res = fuse_ndarray_charges(res, charges[n], charge_types)
  return res


def get_contractable_tensors(R1, R2, cont, dtype, num_charges, DsA, Dscomm,
                             DsB):
  assert R1 >= cont
  assert R2 >= cont
  chargesA = [
      BaseCharge(
          np.random.randint(-5, 5, (num_charges, DsA[n])),
          charge_types=[U1Charge] * num_charges) for n in range(R1 - cont)
  ]
  commoncharges = [
      BaseCharge(
          np.random.randint(-5, 5, (num_charges, Dscomm[n])),
          charge_types=[U1Charge] * num_charges) for n in range(cont)
  ]
  chargesB = [
      BaseCharge(
          np.random.randint(-5, 5, (num_charges, DsB[n])),
          charge_types=[U1Charge] * num_charges) for n in range(R2 - cont)
  ]
  #contracted indices
  indsA = np.random.choice(np.arange(R1), cont, replace=False)
  indsB = np.random.choice(np.arange(R2), cont, replace=False)

  flowsA = np.full(R1, False, dtype=np.bool)
  flowsB = np.full(R2, False, dtype=np.bool)
  flowsB[indsB] = True

  indicesA = [None for _ in range(R1)]
  indicesB = [None for _ in range(R2)]
  for n, ia in enumerate(indsA):
    indicesA[ia] = Index(commoncharges[n], flowsA[ia])
    indicesB[indsB[n]] = Index(commoncharges[n], flowsB[indsB[n]])
  compA = list(set(np.arange(R1)) - set(indsA))
  compB = list(set(np.arange(R2)) - set(indsB))

  for n, ca in enumerate(compA):
    indicesA[ca] = Index(chargesA[n], flowsA[ca])
  for n, cb in enumerate(compB):
    indicesB[cb] = Index(chargesB[n], flowsB[cb])
  indices_final = []
  for n in sorted(compA):
    indices_final.append(indicesA[n])
  for n in sorted(compB):
    indices_final.append(indicesB[n])
  A = BlockSparseTensor.random(indices=indicesA, dtype=dtype)
  B = BlockSparseTensor.random(indices=indicesB, dtype=dtype)
  return A, B, indsA, indsB


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
  with pytest.raises(ValueError):
    _find_best_partition([5])


def test_find_best_partition_raises():
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


@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_compute_num_nonzero(num_charges):
  np.random.seed(12)
  D = 40
  qs = [np.random.randint(-3, 3, (num_charges, D)) for _ in range(3)]
  charges = [BaseCharge(q, charge_types=[U1Charge] * num_charges) for q in qs]
  flows = [False, True, False]
  np_flows = [1, -1, 1]
  fused = fuse_many_ndarray_charges([qs[n] * np_flows[n] for n in range(3)],
                                    [U1Charge] * num_charges)
  nz1 = compute_num_nonzero(charges, flows)
  #pylint: disable=no-member
  nz2 = len(
      np.nonzero(
          np.logical_and.reduce(
              fused.T == np.zeros((1, num_charges), dtype=np.int16),
              axis=1))[0])
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


def test_reduce_charges_2():
  left_charges = np.asarray([[-2, 0, 1, 0, 0], [-3, 0, 2, 1,
                                                0]]).astype(np.int16)
  right_charges = np.asarray([[-1, 0, 2, 1], [-2, 2, 7, 0]]).astype(np.int16)
  target_charge = np.zeros((2, 1), dtype=np.int16)
  fused_charges = fuse_ndarray_charges(left_charges, right_charges,
                                       [U1Charge, U1Charge])
  dense_positions = reduce_charges([
      BaseCharge(left_charges, charge_types=[U1Charge, U1Charge]),
      BaseCharge(right_charges, charge_types=[U1Charge, U1Charge])
  ], [False, False],
                                   target_charge,
                                   return_locations=True)

  np.testing.assert_allclose(dense_positions[0].charges, 0)
  #pylint: disable=no-member
  np.testing.assert_allclose(
      dense_positions[1],
      np.nonzero(
          np.logical_and.reduce(fused_charges.T == target_charge.T, axis=1))[0])


@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_reduce_charges_non_trivial(num_charges):
  np.random.seed(10)
  left_charges = np.random.randint(-5, 6, (num_charges, 200), dtype=np.int16)
  right_charges = np.random.randint(-5, 6, (num_charges, 200), dtype=np.int16)

  target_charge = np.random.randint(-2, 3, (num_charges, 3), dtype=np.int16)
  charge_types = [U1Charge] * num_charges
  fused_charges = fuse_ndarray_charges(left_charges, right_charges,
                                       charge_types)

  dense_positions = reduce_charges([
      BaseCharge(left_charges, charge_types=charge_types),
      BaseCharge(right_charges, charge_types=charge_types)
  ], [False, False],
                                   target_charge,
                                   return_locations=True)
  assert np.all(
      np.isin(
          np.squeeze(dense_positions[0].charges), np.squeeze(target_charge)))
  tmp = []
  #pylint: disable=unsubscriptable-object
  for n in range(target_charge.shape[1]):
    #pylint: disable=no-member
    tmp.append(
        np.logical_and.reduce(
            fused_charges.T == target_charge[:, n][None, :], axis=1))
  #pylint: disable=no-member
  mask = np.logical_or.reduce(tmp)
  np.testing.assert_allclose(dense_positions[1], np.nonzero(mask)[0])


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
  #pylint: disable=no-member
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


def test_ChargeArray_init_raises():
  np.random.seed(10)
  D = 10
  rank = 4
  charges = [U1Charge.random(-5, 5, D) for _ in range(rank)]
  data = np.random.uniform(0, 1, size=D**rank)
  flows = np.random.choice([True, False], size=rank, replace=True)
  order = [[n + 10] for n in range(rank)]
  with pytest.raises(ValueError):
    ChargeArray(data, charges, flows, order=order)


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
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_ChargeArray_todense(dtype, num_charges):
  Ds = [8, 9, 10, 11]
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds[n])),
              charge_types=[U1Charge] * num_charges), False) for n in range(4)
  ]
  arr = ChargeArray.random(indices, dtype=dtype)
  np.testing.assert_allclose(arr.todense(), np.reshape(arr.data, Ds))


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize(
    'Ds', [[[10, 12], [11]], [[8, 9], [10, 11]], [[8, 9], [10, 11], [12, 13]]])
def test_ChargeArray_reshape(dtype, Ds):
  flat_Ds = sum(Ds, [])
  R = len(flat_Ds)
  indices = [Index(U1Charge.random(-5, 5, flat_Ds[n]), False) for n in range(R)]
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
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), False) for n in range(4)]
  arr = ChargeArray.random(indices)
  with pytest.raises(ValueError):
    arr.reshape([64, 65])

  arr2 = arr.reshape([72, 110])
  with pytest.raises(ValueError):
    arr2.reshape([9, 8, 10, 11])

  Ds = [8, 9, 0, 11]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), False) for n in range(4)]
  arr3 = ChargeArray.random(indices)
  with pytest.raises(ValueError):
    arr3.reshape([72, 0])


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


def test_transpose_raises():
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [Index(U1Charge.random(-5, 5, Ds[n]), flows[n]) for n in range(4)]
  arr = ChargeArray.random(indices)
  order = [2, 1, 0]
  with pytest.raises(ValueError):
    arr.transpose(order)


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


@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_transpose_data(num_charges):
  Ds = np.array([8, 9, 10, 11])
  order = [2, 0, 1, 3]
  flows = [True, False, True, False]
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds[n])),
              charge_types=[U1Charge] * num_charges), flows[n])
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


@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_transpose_reshape_transpose_data(num_charges):
  Ds = np.array([8, 9, 10, 11])
  flows = [True, False, True, False]
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds[n])),
              charge_types=[U1Charge] * num_charges), flows[n])
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
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_BlockSparseTensor_random(dtype, num_charges):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds[n])),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(rank)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)

  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_BlockSparseTensor_randn(dtype, num_charges):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds[n])),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(rank)
  ]
  arr = BlockSparseTensor.randn(indices, dtype=dtype)

  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_BlockSparseTensor_ones(dtype, num_charges):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds[n])),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(rank)
  ]
  arr = BlockSparseTensor.ones(indices, dtype=dtype)
  np.testing.assert_allclose(arr.data, 1)
  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])


@pytest.mark.parametrize('dtype', np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_BlockSparseTensor_zeros(dtype, num_charges):
  np.random.seed(10)
  Ds = [8, 9, 10, 11]
  rank = 4
  flows = np.random.choice([True, False], size=rank, replace=True)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds[n])),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(rank)
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


@pytest.mark.parametrize('op', [np.add, np.subtract])
def test_add_sub_raises(op):
  np.random.seed(10)
  Ds1 = [3, 4, 5, 6]
  Ds2 = [4, 5, 6, 7]

  indices1 = [Index(U1Charge.random(-5, 5, Ds1[n]), False) for n in range(4)]
  indices2 = [Index(U1Charge.random(-5, 5, Ds2[n]), False) for n in range(4)]
  a = BlockSparseTensor.randn(indices1)
  b = BlockSparseTensor.randn(indices2)
  with pytest.raises(TypeError):
    op(a, np.array([1, 2, 3]))
  with pytest.raises(ValueError):
    op(a, b)

  Ds3 = [3, 3, 3, 3]
  Ds4 = [9, 9]
  indices3 = [Index(U1Charge.random(-5, 5, Ds3[n]), False) for n in range(4)]
  indices4 = [Index(U1Charge.random(-5, 5, Ds4[n]), False) for n in range(2)]
  c = BlockSparseTensor.randn(indices3).reshape([9, 9])
  d = BlockSparseTensor.randn(indices4)
  with pytest.raises(ValueError):
    op(c, d)

  Ds5 = [200, 200]
  Ds6 = [200, 200]
  indices5 = [Index(U1Charge.random(-5, 5, Ds5[n]), False) for n in range(2)]
  indices6 = [Index(U1Charge.random(-5, 5, Ds6[n]), False) for n in range(2)]
  e = BlockSparseTensor.randn(indices5)
  f = BlockSparseTensor.randn(indices6)
  with pytest.raises(ValueError):
    op(e, f)


@pytest.mark.parametrize('dtype', np_dtypes)
def test_mul(dtype):
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = 5 * a
  np.testing.assert_allclose(b.data, a.data * 5)


def test_mul_raises():
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices)
  with pytest.raises(TypeError):
    [1, 2] * a


@pytest.mark.parametrize('dtype', np_dtypes)
def test_rmul(dtype):
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = a * 5
  np.testing.assert_allclose(b.data, a.data * 5)


def test_rmul_raises():
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices)
  with pytest.raises(TypeError):
    _ = a * np.array([1, 2])


@pytest.mark.parametrize('dtype', np_dtypes)
def test_truediv(dtype):
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices, dtype=dtype)
  b = a / 5
  np.testing.assert_allclose(b.data, a.data / 5)


def test_truediv_raises():
  np.random.seed(10)
  indices = [Index(U1Charge.random(-5, 5, 10), False) for _ in range(4)]
  a = BlockSparseTensor.randn(indices)
  with pytest.raises(TypeError):
    _ = a / np.array([1, 2])


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
@pytest.mark.parametrize('Ds', [[200, 100], [100, 200]])
def test_get_diag(dtype, num_charges, Ds):
  np.random.seed(10)
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
  left, _ = np.divmod(inds, Ds[1])
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
@pytest.mark.parametrize('Ds', [[0, 100], [100, 0]])
def test_get_empty_diag(dtype, num_charges, Ds):
  np.random.seed(10)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-2, 3, (num_charges, Ds[n])),
              charge_types=[U1Charge] * num_charges), False) for n in range(2)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)
  diagonal = diag(arr)
  np.testing.assert_allclose([], diagonal.data)
  for c in diagonal.flat_charges:
    assert len(c) == 0


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
  np.random.seed(10)
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


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize("R1, R2, cont", [(4, 4, 2), (4, 3, 3), (3, 4, 3)])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot(R1, R2, cont, dtype, num_charges):
  np.random.seed(10)
  DsA = np.random.randint(5, 10, R1 - cont)
  Dscomm = np.random.randint(5, 10, cont)
  DsB = np.random.randint(5, 10, R2 - cont)
  A, B, indsA, indsB = get_contractable_tensors(R1, R2, cont, dtype,
                                                num_charges, DsA, Dscomm, DsB)
  res = tensordot(A, B, (indsA, indsB))
  dense_res = np.tensordot(A.todense(), B.todense(), (indsA, indsB))
  np.testing.assert_allclose(dense_res, res.todense())
  free_inds_A = np.sort(list(set(np.arange(len(A.shape))) - set(indsA)))
  free_inds_B = np.sort(list(set(np.arange(len(B.shape))) - set(indsB)))
  for n, fiA in enumerate(free_inds_A):
    assert charge_equal(res.charges[n][0], A.charges[fiA][0])
  for n in range(len(free_inds_A), len(free_inds_A) + len(free_inds_B)):
    assert charge_equal(res.charges[n][0],
                        B.charges[free_inds_B[n - len(free_inds_A)]][0])


@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot_empty_tensors(dtype, num_charges):
  A, B, iA, iB = get_contractable_tensors(
      R1=4,
      R2=4,
      cont=2,
      dtype=dtype,
      num_charges=num_charges,
      DsA=[10, 0],
      Dscomm=[0, 4],
      DsB=[8, 0])
  free_inds_A = np.sort(list(set(np.arange(len(A.shape))) - set(iA)))
  free_inds_B = np.sort(list(set(np.arange(len(B.shape))) - set(iB)))
  res = tensordot(A, B, (iA, iB))
  assert len(res.data) == 0
  for n in range(2):
    assert charge_equal(res.charges[n][0], A.charges[free_inds_A[n]][0])
  for n in range(2, 4):
    assert charge_equal(res.charges[n][0], B.charges[free_inds_B[n - 2]][0])


def test_tensordot_raises():
  R1 = 3
  R2 = 3
  dtype = np.float64
  np.random.seed(10)
  Ds1 = np.arange(2, 2 + R1)
  Ds2 = np.arange(2 + R1, 2 + R1 + R2)
  is1 = [Index(U1Charge.random(-5, 5, Ds1[n]), False) for n in range(R1)]
  is2 = [Index(U1Charge.random(-5, 5, Ds2[n]), False) for n in range(R2)]
  A = BlockSparseTensor.random(is1, dtype=dtype)
  B = BlockSparseTensor.random(is2, dtype=dtype)

  with pytest.raises(ValueError):
    tensordot(A, B, ([0], [1, 2]))
  with pytest.raises(ValueError):
    tensordot(A, B, [0, [1, 2]])
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 0], [1, 2]))
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 1], [1, 1]))
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 4], [1, 2]))
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 4], [1, 4]))
  with pytest.raises(ValueError):
    tensordot(A, B, ([0, 1], [0, 1]))
  with pytest.raises(ValueError):
    tensordot(A, A, ([0, 1], [0, 1]))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot_reshape(dtype, num_charges):
  np.random.seed(10)
  R1 = 4
  R2 = 4

  q = np.random.randint(-5, 5, (num_charges, 10), dtype=np.int16)
  charges1 = [
      BaseCharge(q, charge_types=[U1Charge] * num_charges) for n in range(R1)
  ]
  charges2 = [
      BaseCharge(q, charge_types=[U1Charge] * num_charges) for n in range(R2)
  ]
  flowsA = np.asarray([False] * R1)
  flowsB = np.asarray([True] * R2)
  A = BlockSparseTensor.random(
      indices=[Index(charges1[n], flowsA[n]) for n in range(R1)], dtype=dtype)
  B = BlockSparseTensor.random(
      indices=[Index(charges2[n], flowsB[n]) for n in range(R2)], dtype=dtype)

  Adense = A.todense().reshape((10, 10 * 10, 10))
  Bdense = B.todense().reshape((10 * 10, 10, 10))

  A = A.reshape((10, 10 * 10, 10))
  B = B.reshape((10 * 10, 10, 10))

  res = tensordot(A, B, ([0, 1], [2, 0]))
  dense = np.tensordot(Adense, Bdense, ([0, 1], [2, 0]))
  np.testing.assert_allclose(dense, res.todense())


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R1, R2", [(2, 2), (3, 3), (4, 4), (1, 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot_inner(R1, R2, dtype, num_charges):
  np.random.seed(10)
  DsA = np.random.randint(3, 5, R1)
  Dscomm = np.random.randint(3, 5, 0)
  DsB = np.random.randint(3, 5, R2)
  A, B, indsA, indsB = get_contractable_tensors(R1, R2, 0, dtype, num_charges,
                                                DsA, Dscomm, DsB)
  res = tensordot(A, B, (indsA, indsB))
  dense_res = np.tensordot(A.todense(), B.todense(), (indsA, indsB))
  np.testing.assert_allclose(dense_res, res.todense())


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R1, R2", [(2, 2), (3, 3), (4, 4), (1, 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot_inner_transpose(R1, R2, dtype, num_charges):
  np.random.seed(10)
  DsA = np.random.randint(3, 5, R1)
  Dscomm = np.random.randint(3, 5, 0)
  DsB = np.random.randint(3, 5, R2)
  A, B, indsA, indsB = get_contractable_tensors(R1, R2, 0, dtype, num_charges,
                                                DsA, Dscomm, DsB)
  orderA = np.arange(R1)
  orderB = np.arange(R2)
  np.random.shuffle(orderA)
  np.random.shuffle(orderB)
  A_ = A.transpose(orderA)
  B_ = B.transpose(orderB)
  _, indposA = np.unique(orderA, return_index=True)
  _, indposB = np.unique(orderB, return_index=True)
  indsA_ = indposA[indsA]
  indsB_ = indposB[indsB]
  res = tensordot(A_, B_, (indsA_, indsB_))
  dense_res = np.tensordot(A_.todense(), B_.todense(), (indsA_, indsB_))
  np.testing.assert_allclose(dense_res, res.todense())


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("R1, R2", [(2, 2), (2, 1), (1, 2), (1, 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_tensordot_outer(R1, R2, dtype, num_charges):
  np.random.seed(10)
  DsA = np.random.randint(3, 5, R1)
  Dscomm = np.random.randint(3, 5, 0)
  DsB = np.random.randint(3, 5, R2)
  A, B, _, _ = get_contractable_tensors(R1, R2, 0, dtype, num_charges, DsA,
                                        Dscomm, DsB)
  res = tensordot(A, B, axes=0)
  dense_res = np.tensordot(A.todense(), B.todense(), axes=0)
  np.testing.assert_allclose(dense_res, res.todense())
  for n in range(R1):
    assert charge_equal(res.charges[n][0], A.charges[n][0])
  for n in range(R1, R1 + R2):
    assert charge_equal(res.charges[n][0], B.charges[n - R1][0])


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3, 4])
def test_matmul(dtype, num_charges):
  np.random.seed(10)
  Ds1 = [100, 200]
  is1 = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds1[n]), dtype=np.int16),
              charge_types=[U1Charge] * num_charges), False) for n in range(2)
  ]
  is2 = [
      is1[1].copy().flip_flow(),
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, 150), dtype=np.int16),
              charge_types=[U1Charge] * num_charges), False)
  ]
  tensor1 = BlockSparseTensor.random(is1, dtype=dtype)
  tensor2 = BlockSparseTensor.random(is2, dtype=dtype)
  result = tensor1 @ tensor2
  assert result.dtype == dtype
  dense_result = tensor1.todense() @ tensor2.todense()
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


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("Ds, R1", [([20, 21], 1), ([18, 19, 20], 2),
                                    ([18, 19, 20], 1), ([0, 10], 1),
                                    ([10, 0], 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_svd_prod(dtype, Ds, R1, num_charges):
  np.random.seed(10)
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (num_charges, Ds[n])),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  d1 = np.prod(Ds[:R1])
  d2 = np.prod(Ds[R1:])
  A = A.reshape([d1, d2])

  U, S, V = svd(A, full_matrices=False)
  A_ = U @ diag(S) @ V
  assert A_.dtype == A.dtype
  np.testing.assert_allclose(A.data, A_.data)
  for n in range(len(A._charges)):
    assert charge_equal(A_._charges[n], A._charges[n])


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("Ds, R1", [([20, 21], 1), ([18, 19, 20], 2),
                                    ([18, 19, 20], 1), ([0, 10], 1),
                                    ([10, 0], 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_svd_singvals(dtype, Ds, R1, num_charges):
  np.random.seed(10)
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (num_charges, Ds[n])),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)

  d1 = np.prod(Ds[:R1])
  d2 = np.prod(Ds[R1:])
  A = A.reshape([d1, d2])
  _, S1, _ = svd(A, full_matrices=False)
  S2 = svd(A, full_matrices=False, compute_uv=False)
  np.testing.assert_allclose(S1.data, S2.data)
  Sdense = np.linalg.svd(A.todense(), compute_uv=False)
  np.testing.assert_allclose(
      np.sort(Sdense[Sdense > 1E-15]), np.sort(S2.data[S2.data > 0.0]))


@pytest.mark.parametrize("mode", ['complete', 'reduced'])
@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("Ds, R1", [([20, 21], 1), ([18, 19, 20], 2),
                                    ([18, 19, 20], 1), ([10, 0], 1),
                                    ([0, 10], 1)])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_qr_prod(dtype, Ds, R1, mode, num_charges):
  np.random.seed(10)
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (num_charges, Ds[n])),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  d1 = np.prod(Ds[:R1])
  d2 = np.prod(Ds[R1:])
  A = A.reshape([d1, d2])
  Q, R = qr(A, mode=mode)
  A_ = Q @ R
  assert A_.dtype == A.dtype
  np.testing.assert_allclose(A.data, A_.data)
  for n in range(len(A._charges)):
    assert charge_equal(A_._charges[n], A._charges[n])


def test_qr_raises():
  np.random.seed(10)
  dtype = np.float64
  num_charges = 1
  Ds = [20, 21]
  R1 = 1
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (num_charges, Ds[n])),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [True] * R
  A = BlockSparseTensor.random([Index(charges[n], flows[n]) for n in range(R)],
                               dtype=dtype)
  d1 = np.prod(Ds[:R1])
  d2 = np.prod(Ds[R1:])
  A = A.reshape([d1, d2])
  with pytest.raises(ValueError):
    qr(A, mode='fake_mode')


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("Ds", [[20], [9, 10], [6, 7, 8], [0]])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_eigh_prod(dtype, Ds, num_charges):
  np.random.seed(10)
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (num_charges, Ds[n]), dtype=np.int16),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [False] * R
  inds = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.random(
      inds + [i.copy().flip_flow() for i in inds], dtype=dtype)
  dims = np.prod(Ds)
  A = A.reshape([dims, dims])
  B = A + A.T.conj()
  E, V = eigh(B)
  B_ = V @ diag(E) @ V.conj().T
  np.testing.assert_allclose(B.data, B_.data)
  for n in range(len(B._charges)):
    assert charge_equal(B_._charges[n], B._charges[n])


def test_eigh_raises():
  np.random.seed(10)
  num_charges = 1
  D = 20
  R = 3
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (num_charges, D), dtype=np.int16),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [False] * R
  inds = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.random(inds)
  with pytest.raises(NotImplementedError):
    eigh(A)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_inv(dtype, num_charges):
  np.random.seed(10)
  R = 2
  D = 10
  charge = BaseCharge(
      np.random.randint(-5, 6, (num_charges, D), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  flows = [True, False]
  A = BlockSparseTensor.random([Index(charge, flows[n]) for n in range(R)],
                               (-0.5, 0.5),
                               dtype=dtype)
  invA = inv(A)
  left_eye = invA @ A

  blocks, _, shapes = _find_diagonal_sparse_blocks(left_eye.flat_charges,
                                                   left_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(left_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12

  right_eye = A @ invA
  blocks, _, shapes = _find_diagonal_sparse_blocks(right_eye.flat_charges,
                                                   right_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(right_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12


def test_inv_raises():
  num_charges = 1
  np.random.seed(10)
  R = 3
  D = 10
  charge = BaseCharge(
      np.random.randint(-5, 6, (num_charges, D), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  A = BlockSparseTensor.random([Index(charge, False) for n in range(R)],
                               (-0.5, 0.5))
  with pytest.raises(ValueError):
    inv(A)


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize("Ds", [[20], [9, 10], [6, 7, 8], [0]])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_eig_prod(dtype, Ds, num_charges):
  np.random.seed(10)
  R = len(Ds)
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (num_charges, Ds[n]), dtype=np.int16),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [False] * R
  inds = [Index(charges[n], flows[n]) for n in range(R)]

  A = BlockSparseTensor.random(
      inds + [i.copy().flip_flow() for i in inds], dtype=dtype)
  dims = np.prod(Ds)
  A = A.reshape([dims, dims])
  E, V = eig(A)
  A_ = V @ diag(E) @ inv(V)
  np.testing.assert_allclose(A.data, A_.data)


def test_eig_raises():
  np.random.seed(10)
  num_charges = 1
  D = 20
  R = 3
  charges = [
      BaseCharge(
          np.random.randint(-5, 6, (num_charges, D), dtype=np.int16),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = [False] * R
  inds = [Index(charges[n], flows[n]) for n in range(R)]
  A = BlockSparseTensor.random(inds)
  with pytest.raises(NotImplementedError):
    eig(A)


#Note the case num_charges=4 is most likely testing  empty tensors
@pytest.mark.parametrize("dtype", np_tensordot_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize("Ds", [[20], [9, 10], [6, 7, 8], [9, 8, 0, 10]])
def test_sqrt(dtype, num_charges, Ds):
  np.random.seed(10)
  R = len(Ds)
  flows = np.random.choice([True, False], replace=True, size=R)
  indices = [
      Index(
          BaseCharge(
              np.random.randint(-5, 6, (num_charges, Ds[n]), dtype=np.int16),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(R)
  ]
  arr = BlockSparseTensor.random(indices, dtype=dtype)
  sqrtarr = sqrt(arr)
  np.testing.assert_allclose(sqrtarr.data, np.sqrt(arr.data))


@pytest.mark.parametrize("dtype", np_dtypes)
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('D', [0, 10])
def test_eye(dtype, num_charges, D):
  charge = BaseCharge(
      np.random.randint(-5, 6, (num_charges, D), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  flow = False
  index = Index(charge, flow)
  A = eye(index, dtype=dtype)
  blocks, _, shapes = _find_diagonal_sparse_blocks(A.flat_charges, A.flat_flows,
                                                   1)
  for n, block in enumerate(blocks):
    t = np.reshape(A.data[block], shapes[:, n])
    np.testing.assert_almost_equal(t, np.eye(t.shape[0], t.shape[1]))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('D', [0, 100])
def test_trace_matrix(dtype, num_charges, D):
  np.random.seed(10)
  R = 2
  charge = BaseCharge(
      np.random.randint(-5, 6, (num_charges, D), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  flows = [True, False]
  matrix = BlockSparseTensor.random([Index(charge, flows[n]) for n in range(R)],
                                    dtype=dtype)
  res = trace(matrix)
  res_dense = np.trace(matrix.todense())
  np.testing.assert_allclose(res, res_dense)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
@pytest.mark.parametrize('D1, D2', [(10, 12), (0, 10)])
def test_trace_tensor(dtype, num_charges, D1, D2):
  np.random.seed(10)
  charge1 = BaseCharge(
      np.random.randint(-5, 6, (num_charges, D1), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  charge2 = BaseCharge(
      np.random.randint(-5, 6, (num_charges, D2), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  indices = [Index(charge1, False), Index(charge2, False), Index(charge1, True)]
  tensor = BlockSparseTensor.random(indices, dtype=dtype)
  res = trace(tensor, (0, 2))
  assert res.sparse_shape[0] == indices[1]
  res_dense = np.trace(tensor.todense(), axis1=0, axis2=2)
  np.testing.assert_allclose(res.todense(), res_dense)


@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_trace_raises(num_charges):
  np.random.seed(10)
  D = 20
  charge1 = BaseCharge(
      np.random.randint(-5, 6, (num_charges, D), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  A1 = BlockSparseTensor.random([Index(charge1, False)])
  with pytest.raises(ValueError):
    trace(A1)

  charge2 = BaseCharge(
      np.random.randint(-5, 6, (num_charges, D + 1), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  indices = [
      Index(charge1, False),
      Index(charge2, False),
      Index(charge1, False)
  ]
  A2 = BlockSparseTensor.random(indices)
  with pytest.raises(ValueError):
    trace(A2, axes=(0, 1))
  with pytest.raises(ValueError):
    trace(A2, axes=(0, 2))
  with pytest.raises(ValueError):
    trace(A2, axes=(0, 1, 2))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_pinv(dtype, num_charges):
  np.random.seed(10)
  R = 2
  D = 10
  charge = BaseCharge(
      np.random.randint(-5, 6, (num_charges, D), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  flows = [True, False]
  A = BlockSparseTensor.random([Index(charge, flows[n]) for n in range(R)],
                               (-0.5, 0.5),
                               dtype=dtype)
  invA = pinv(A)
  left_eye = invA @ A

  blocks, _, shapes = _find_diagonal_sparse_blocks(left_eye.flat_charges,
                                                   left_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(left_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12

  right_eye = A @ invA
  blocks, _, shapes = _find_diagonal_sparse_blocks(right_eye.flat_charges,
                                                   right_eye.flat_flows, 1)
  for n, block in enumerate(blocks):
    t = np.reshape(right_eye.data[block], shapes[:, n])
    assert np.linalg.norm(t - np.eye(t.shape[0], t.shape[1])) < 1E-12


def test_pinv_raises():
  num_charges = 1
  np.random.seed(10)
  R = 3
  D = 10
  charge = BaseCharge(
      np.random.randint(-5, 6, (num_charges, D), dtype=np.int16),
      charge_types=[U1Charge] * num_charges)
  A = BlockSparseTensor.random([Index(charge, False) for n in range(R)],
                               (-0.5, 0.5))
  with pytest.raises(ValueError):
    pinv(A)


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
              np.random.randint(-5, 6, (num_charges, Ds[n])),
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
              np.random.randint(-5, 6, (num_charges, Ds[n])),
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
              np.random.randint(-5, 6, (num_charges, Ds[n])),
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
              np.random.randint(-5, 6, (num_charges, Ds[n])),
              charge_types=[U1Charge] * num_charges), flows[n])
      for n in range(rank)
  ]
  arr = randn(indices, dtype=dtype)

  np.testing.assert_allclose(Ds, arr.shape)
  np.testing.assert_allclose(arr.flat_flows, flows)
  for n in range(4):
    assert charge_equal(arr.charges[n][0], indices[n].flat_charges[0])
