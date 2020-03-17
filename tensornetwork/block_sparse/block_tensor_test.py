import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_sparse.charge import U1Charge, fuse_charges, charge_equal, fuse_ndarrays, fuse_ndarray_charges, BaseCharge
from tensornetwork.block_sparse.index import Index
# pylint: disable=line-too-long
from tensornetwork.block_sparse.block_tensor import flatten, get_flat_meta_data, fuse_stride_arrays, compute_sparse_lookup, _find_best_partition, compute_fused_charge_degeneracies, compute_unique_fused_charges, compute_num_nonzero, reduce_charges, _find_diagonal_sparse_blocks

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


@pytest.mark.parametrize('num_legs', [2, 3, 4])
@pytest.mark.parametrize('num_charges', [1, 2, 3])
def test_find_transposed_diagonal_sparse_blocks(num_legs):
  np.random.seed(10)
  order = np.arange(num_legs)
  while np.all(order == np.arange(num_legs)):
    np.random.shuffle(order)

  np_charges = [
      np.random.randint(-5, 5, (num_charges, 60), dtype=np.int16)
      for _ in range(num_legs)
  ]
  fused = np.stack([
      fuse_ndarrays([np_charges[order[n]][c, :]
                     for n in range(num_legs)])
      for c in range(num_charges)
  ],
                   axis=0)

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
  nz = np.nonzero(
      np.logical_and.reduce(fused.T == np.zeros((1, num_charges)), axis=1))[0]
