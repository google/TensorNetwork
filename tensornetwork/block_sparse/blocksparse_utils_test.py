import numpy as np
import pytest
import itertools
from tensornetwork.block_sparse.charge import (U1Charge, charge_equal,
                                               fuse_ndarray_charges, BaseCharge)
from tensornetwork.block_sparse.utils import (fuse_ndarrays, _get_strides,
                                              fuse_stride_arrays, unique)
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparse_utils import (
    compute_sparse_lookup, compute_fused_charge_degeneracies,
    compute_unique_fused_charges, compute_num_nonzero, reduce_charges,
    _find_diagonal_sparse_blocks, _find_transposed_diagonal_sparse_blocks,
    get_flat_meta_data, _to_string)

np_dtypes = [np.float64, np.complex128]
np_tensordot_dtypes = [np.float64, np.complex128]


def fuse_many_ndarray_charges(charges, charge_types):
  res = fuse_ndarray_charges(charges[0], charges[1], charge_types)
  for n in range(2, len(charges)):
    res = fuse_ndarray_charges(res, charges[n], charge_types)
  return res


def test_flat_meta_data():
  i1 = Index([
      U1Charge.random(dimension=20, minval=-2, maxval=2),
      U1Charge.random(dimension=20, minval=-2, maxval=2)
  ],
             flow=[True, False])

  i2 = Index([
      U1Charge.random(dimension=20, minval=-2, maxval=2),
      U1Charge.random(dimension=20, minval=-2, maxval=2)
  ],
             flow=[False, True])
  expected_charges = [
      i1._charges[0], i1._charges[1], i2._charges[0], i2._charges[1]
  ]
  expected_flows = [True, False, False, True]
  charges, flows = get_flat_meta_data([i1, i2])
  np.testing.assert_allclose(flows, expected_flows)
  for n, c in enumerate(charges):
    assert charge_equal(c, expected_charges[n])


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
  lookup, unique_, labels = compute_sparse_lookup(charges, flows, targets)
  np.testing.assert_allclose(lookup, expected_lookup)
  np.testing.assert_allclose(expected_unique, np.squeeze(unique_.charges))
  np.testing.assert_allclose(labels, expected_labels_to_unique)


@pytest.mark.parametrize('flow', [True, False])
def test_compute_sparse_lookup_non_ordered(flow):
  np_flow = np.int(-(np.int(flow) - 0.5) * 2)
  charge_labels = np.array([0, 0, 1, 5, 5, 0, 2, 3, 2, 3, 4, 0, 3, 3, 1, 5])
  unique_charges = np.array([-1, 0, 1, -5, 7, 2])
  np_targets = np.array([-1, 0, 2])
  charges = [U1Charge(unique_charges, charge_labels=charge_labels)]
  inds = np.nonzero(
      np.isin((np_flow * unique_charges)[charge_labels], np_targets))[0]
  targets = U1Charge(np_targets)
  lookup, unique_, labels = compute_sparse_lookup(charges, [flow], targets)
  np.testing.assert_allclose(labels, np.sort(labels))
  np.testing.assert_allclose(
      np.squeeze(unique_.charges[lookup, :]),
      (np_flow * unique_charges)[charge_labels][inds])


def test_compute_fused_charge_degeneracies():
  np.random.seed(10)
  qs = [np.random.randint(-3, 3, 100) for _ in range(3)]
  charges = [U1Charge(q) for q in qs]
  flows = [False, True, False]
  np_flows = [1, -1, 1]
  unique_, degens = compute_fused_charge_degeneracies(charges, flows)
  fused = fuse_ndarrays([qs[n] * np_flows[n] for n in range(3)])
  exp_unique, exp_degens = unique(fused, return_counts=True)
  np.testing.assert_allclose(np.squeeze(unique_.charges), exp_unique)
  np.testing.assert_allclose(degens, exp_degens)


def test_compute_unique_fused_charges():
  np.random.seed(10)
  qs = [np.random.randint(-3, 3, 100) for _ in range(3)]
  charges = [U1Charge(q) for q in qs]
  flows = [False, True, False]
  np_flows = [1, -1, 1]
  unique_ = compute_unique_fused_charges(charges, flows)
  fused = fuse_ndarrays([qs[n] * np_flows[n] for n in range(3)])
  exp_unique = unique(fused)
  np.testing.assert_allclose(np.squeeze(unique_.charges), exp_unique)


@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_compute_num_nonzero(num_charges):
  np.random.seed(12)
  D = 40
  qs = [np.random.randint(-3, 3, (D, num_charges)) for _ in range(3)]
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
              fused == np.zeros((1, num_charges), dtype=np.int16), axis=1))[0])
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
                                                0]]).astype(np.int16).T
  right_charges = np.asarray([[-1, 0, 2, 1], [-2, 2, 7, 0]]).astype(np.int16).T
  target_charge = np.zeros((1, 2), dtype=np.int16)
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
      np.nonzero(np.logical_and.reduce(fused_charges == target_charge,
                                       axis=1))[0])


@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_reduce_charges_non_trivial(num_charges):
  np.random.seed(10)
  left_charges = np.random.randint(-5, 6, (200, num_charges), dtype=np.int16)
  right_charges = np.random.randint(-5, 6, (200, num_charges), dtype=np.int16)

  target_charge = np.random.randint(-2, 3, (3, num_charges), dtype=np.int16)
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
  for n in range(target_charge.shape[0]):
    #pylint: disable=no-member
    tmp.append(
        np.logical_and.reduce(
            fused_charges == target_charge[n, :][None, :], axis=1))
  #pylint: disable=no-member
  mask = np.logical_or.reduce(tmp)
  np.testing.assert_allclose(dense_positions[1], np.nonzero(mask)[0])


@pytest.mark.parametrize('num_legs', [2, 3, 4])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_find_diagonal_sparse_blocks(num_legs, num_charges):
  np.random.seed(10)
  np_charges = [
      np.random.randint(-5, 5, (60, num_charges), dtype=np.int16)
      for _ in range(num_legs)
  ]
  fused = np.stack([
      fuse_ndarrays([np_charges[n][:, c]
                     for n in range(num_legs)])
      for c in range(num_charges)
  ],
                   axis=1)

  left_charges = np.stack([
      fuse_ndarrays([np_charges[n][:, c]
                     for n in range(num_legs // 2)])
      for c in range(num_charges)
  ],
                          axis=1)
  right_charges = np.stack([
      fuse_ndarrays(
          [np_charges[n][:, c]
           for n in range(num_legs // 2, num_legs)])
      for c in range(num_charges)
  ],
                           axis=1)
  #pylint: disable=no-member
  nz = np.nonzero(
      np.logical_and.reduce(fused == np.zeros((1, num_charges)), axis=1))[0]
  linear_locs = np.arange(len(nz))
  # pylint: disable=no-member
  left_inds, _ = np.divmod(nz, right_charges.shape[0])
  left = left_charges[left_inds, :]
  unique_left = unique(left)
  blocks = []
  for n in range(unique_left.shape[0]):
    ul = unique_left[n, :][None, :]
    #pylint: disable=no-member
    blocks.append(linear_locs[np.nonzero(
        np.logical_and.reduce(left == ul, axis=1))[0]])

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
      np.random.randint(-5, 5, (D, num_charges), dtype=np.int16)
      for _ in range(num_legs)
  ]
  tr_charge_list = []
  charge_list = []
  for c in range(num_charges):

    tr_charge_list.append(
        fuse_ndarrays([np_charges[order[n]][:, c] for n in range(num_legs)]))
    charge_list.append(
        fuse_ndarrays([np_charges[n][:, c] for n in range(num_legs)]))

  tr_fused = np.stack(tr_charge_list, axis=1)
  fused = np.stack(charge_list, axis=1)

  dims = [c.shape[0] for c in np_charges]
  strides = _get_strides(dims)
  transposed_linear_positions = fuse_stride_arrays(dims,
                                                   [strides[o] for o in order])
  left_charges = np.stack([
      fuse_ndarrays([np_charges[order[n]][:, c]
                     for n in range(num_legs // 2)])
      for c in range(num_charges)
  ],
                          axis=1)
  right_charges = np.stack([
      fuse_ndarrays(
          [np_charges[order[n]][:, c]
           for n in range(num_legs // 2, num_legs)])
      for c in range(num_charges)
  ],
                           axis=1)
  #pylint: disable=no-member
  mask = np.logical_and.reduce(fused == np.zeros((1, num_charges)), axis=1)
  nz = np.nonzero(mask)[0]
  dense_to_sparse = np.empty(len(mask), dtype=np.int64)
  dense_to_sparse[mask] = np.arange(len(nz))
  #pylint: disable=no-member
  tr_mask = np.logical_and.reduce(
      tr_fused == np.zeros((1, num_charges)), axis=1)
  tr_nz = np.nonzero(tr_mask)[0]
  tr_linear_locs = transposed_linear_positions[tr_nz]
  # pylint: disable=no-member
  left_inds, _ = np.divmod(tr_nz, right_charges.shape[0])
  left = left_charges[left_inds, :]
  unique_left = unique(left)
  blocks = []
  for n in range(unique_left.shape[0]):
    ul = unique_left[n, :][None, :]
    #pylint: disable=no-member
    blocks.append(dense_to_sparse[tr_linear_locs[np.nonzero(
        np.logical_and.reduce(left == ul, axis=1))[0]]])

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


def test_to_string():
  R = 5
  D = 100
  np.random.seed(10)
  cs = [U1Charge.random(D, -5, 5) for _ in range(R)]
  flows = np.random.choice([True, False], size=R, replace=True)
  tr_partition = 3
  order = list(np.random.choice(np.arange(R), size=R, replace=False))
  actual = _to_string(cs, flows, tr_partition, order)
  expected = ''.join([str(c.charges.tostring()) for c in cs] + [
      str(np.array(flows).tostring()),
      str(tr_partition),
      str(np.array(order, dtype=np.int16).tostring())
  ])
  assert actual == expected
