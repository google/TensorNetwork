import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_tensor.block_tensor import BlockSparseTensor, compute_num_nonzero, compute_dense_to_sparse_mapping, find_sparse_positions, find_dense_positions, map_to_integer
from index import Index, fuse_charges

np_dtypes = [np.float32, np.float16, np.float64, np.complex64, np.complex128]


@pytest.mark.parametrize("dtype", np_dtypes)
def test_block_sparse_init(dtype):
  D = 10  #bond dimension
  B = 10  #number of blocks
  rank = 4
  flows = np.asarray([1 for _ in range(rank)])
  flows[-2::] = -1
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(rank)
  ]
  indices = [
      Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
      for n in range(rank)
  ]
  num_elements = compute_num_nonzero([i.charges for i in indices],
                                     [i.flow for i in indices])
  A = BlockSparseTensor.random(indices=indices, dtype=dtype)
  assert A.dtype == dtype
  for r in range(rank):
    assert A.indices[r].name == 'index{}'.format(r)
  assert A.dense_shape == tuple([D] * rank)
  assert len(A.data) == num_elements


def test_dense_to_sparse_table():
  D = 30  #bond dimension
  B = 4  #number of blocks
  dtype = np.int16  #the dtype of the quantum numbers
  rank = 4
  flows = np.asarray([1 for _ in range(rank)])
  flows[-2::] = -1
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
      for _ in range(rank)
  ]
  num_non_zero = compute_num_nonzero(charges, flows)

  inds = compute_dense_to_sparse_mapping(
      charges=charges, flows=flows, target_charge=0)
  total = np.zeros(len(inds[0]), dtype=np.int16)
  for n in range(len(charges)):
    total += flows[n] * charges[n][inds[n]]

  np.testing.assert_allclose(total, 0)
  assert len(total) == num_non_zero


def test_find_dense_positions():
  left_charges = [-2, 0, 1, 0, 0]
  right_charges = [-1, 0, 2, 1]
  target_charge = 0
  fused_charges = fuse_charges([left_charges, right_charges], [1, 1])
  blocks = find_dense_positions(left_charges, 1, right_charges, 1,
                                target_charge)
  np.testing.assert_allclose(blocks[(-2, 2)], [2])
  np.testing.assert_allclose(blocks[(0, 0)], [5, 13, 17])
  np.testing.assert_allclose(blocks[(1, -1)], [8])


def test_find_dense_positions_2():
  D = 40  #bond dimension
  B = 4  #number of blocks
  dtype = np.int16  #the dtype of the quantum numbers
  rank = 4
  flows = np.asarray([1 for _ in range(rank)])
  flows[-2::] = -1
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
      for _ in range(rank)
  ]
  indices = [
      Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
      for n in range(rank)
  ]
  n1 = compute_num_nonzero([i.charges for i in indices],
                           [i.flow for i in indices])
  row_charges = fuse_charges([indices[n].charges for n in range(rank // 2)],
                             [1 for _ in range(rank // 2)])
  column_charges = fuse_charges(
      [indices[n].charges for n in range(rank // 2, rank)],
      [1 for _ in range(rank // 2, rank)])

  i01 = indices[0] * indices[1]
  i23 = indices[2] * indices[3]
  blocks = find_dense_positions(i01.charges, 1, i23.charges, 1, 0)
  assert sum([len(v) for v in blocks.values()]) == n1

  tensor = BlockSparseTensor.random(indices=indices, dtype=np.float64)
  tensor.reshape((D * D, D * D))
  blocks_2 = tensor.get_diagonal_blocks(return_data=False)
  np.testing.assert_allclose([k[0] for k in blocks.keys()],
                             list(blocks_2.keys()))
  for c in blocks.keys():
    assert np.prod(blocks_2[c[0]][1]) == len(blocks[c])


def test_find_sparse_positions():
  D = 40  #bond dimension
  B = 4  #number of blocks
  dtype = np.int16  #the dtype of the quantum numbers
  rank = 4
  flows = np.asarray([1 for _ in range(rank)])
  flows[-2::] = -1
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
      for _ in range(rank)
  ]
  indices = [
      Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
      for n in range(rank)
  ]
  n1 = compute_num_nonzero([i.charges for i in indices],
                           [i.flow for i in indices])
  row_charges = fuse_charges([indices[n].charges for n in range(rank // 2)],
                             [1 for _ in range(rank // 2)])
  column_charges = fuse_charges(
      [indices[n].charges for n in range(rank // 2, rank)],
      [1 for _ in range(rank // 2, rank)])

  i01 = indices[0] * indices[1]
  i23 = indices[2] * indices[3]
  unique_row_charges = np.unique(i01.charges)
  unique_column_charges = np.unique(i23.charges)
  common_charges = np.intersect1d(
      unique_row_charges, -unique_column_charges, assume_unique=True)
  blocks = find_sparse_positions(
      i01.charges, 1, i23.charges, 1, target_charges=[0])
  assert sum([len(v) for v in blocks.values()]) == n1
  np.testing.assert_allclose(np.sort(blocks[0]), np.arange(n1))


def test_find_sparse_positions_2():
  D = 40  #bond dimension
  B = 4  #number of blocks
  dtype = np.int16  #the dtype of the quantum numbers
  flows = [1, -1]

  rank = len(flows)
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
      for _ in range(rank)
  ]
  indices = [
      Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
      for n in range(rank)
  ]
  i1, i2 = indices
  common_charges = np.intersect1d(i1.charges, i2.charges)
  row_locations = find_sparse_positions(
      left_charges=i1.charges,
      left_flow=flows[0],
      right_charges=i2.charges,
      right_flow=flows[1],
      target_charges=common_charges)
  fused = (i1 * i2).charges
  relevant = fused[np.isin(fused, common_charges)]
  for k, v in row_locations.items():
    np.testing.assert_allclose(np.nonzero(relevant == k)[0], np.sort(v))


def test_map_to_integer():
  dims = [4, 3, 2]
  dim_prod = [6, 2, 1]
  N = 10
  table = np.stack([np.random.randint(0, d, N) for d in dims], axis=1)
  integers = map_to_integer(dims, table)
  ints = []
  for n in range(N):
    i = 0
    for d in range(len(dims)):
      i += dim_prod[d] * table[n, d]
    ints.append(i)
  np.testing.assert_allclose(ints, integers)


def test_ge_diagonal_blocks():
  D = 40  #bond dimension
  B = 4  #number of blocks
  dtype = np.int16  #the dtype of the quantum numbers
  rank = 4
  flows = np.asarray([1 for _ in range(rank)])
  flows[-2::] = -1
  charges = [
      np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
      for _ in range(rank)
  ]
  indices = [
      Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
      for n in range(rank)
  ]
  common_charges = np.intersect1d(indices[0].charges, indices[1].charges)
  row_locations = find_sparse_positions(
      left_charges=indices[0].charges,
      left_flow=1,
      right_charges=indices[1].charges,
      right_flow=1,
      target_charges=common_charges)
