import numpy as np
import pytest

from tensornetwork.block_tensor.charge import U1Charge
from tensornetwork.block_tensor.block_tensor_new import find_diagonal_sparse_blocks, compute_num_nonzero

np_dtypes = [np.float32, np.float16, np.float64, np.complex64, np.complex128]


def test_consistency():
  B = 5
  D = 100
  rank = 4
  qs = [[
      np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
      for _ in range(2)
  ]
        for _ in range(rank)]
  charges1 = [U1Charge(qs[n]) for n in range(rank)]
  charges2 = [ChargeCollection([charges1[n]]) for n in range(rank)]
  charges3 = [
      ChargeCollection([U1Charge([q for q in qs[n]])]) for n in range(rank)
  ]
  flows = [1, 1, 1, -1]
  n1 = compute_num_nonzero(charges1, flows)
  n2 = compute_num_nonzero(charges2, flows)
  n3 = compute_num_nonzero(charges3, flows)
  assert n1 == n2
  assert n1 == n3


# @pytest.mark.parametrize("dtype", np_dtypes)
# def test_block_sparse_init(dtype):
#   D = 10  #bond dimension
#   B = 10  #number of blocks
#   rank = 4
#   flows = np.asarray([1 for _ in range(rank)])
#   flows[-2::] = -1
#   charges = [
#       np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16)
#       for _ in range(rank)
#   ]
#   indices = [
#       Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
#       for n in range(rank)
#   ]
#   num_elements = compute_num_nonzero([i.charges for i in indices],
#                                      [i.flow for i in indices])
#   A = BlockSparseTensor.random(indices=indices, dtype=dtype)
#   assert A.dtype == dtype
#   for r in range(rank):
#     assert A.indices[r].name == 'index{}'.format(r)
#   assert A.dense_shape == tuple([D] * rank)
#   assert len(A.data) == num_elements

# def test_find_dense_positions():
#   left_charges = np.asarray([-2, 0, 1, 0, 0]).astype(np.int16)
#   right_charges = np.asarray([-1, 0, 2, 1]).astype(np.int16)
#   target_charge = 0
#   fused_charges = fuse_charges([left_charges, right_charges], [1, 1])
#   dense_positions = find_dense_positions(left_charges, 1, right_charges, 1,
#                                          target_charge)
#   np.testing.assert_allclose(dense_positions,
#                              np.nonzero(fused_charges == target_charge)[0])

# def test_find_dense_positions_2():
#   D = 40  #bond dimension
#   B = 4  #number of blocks
#   dtype = np.int16  #the dtype of the quantum numbers
#   rank = 4
#   flows = np.asarray([1 for _ in range(rank)])
#   flows[-2::] = -1
#   charges = [
#       np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#       for _ in range(rank)
#   ]
#   indices = [
#       Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
#       for n in range(rank)
#   ]
#   n1 = compute_num_nonzero([i.charges for i in indices],
#                            [i.flow for i in indices])
#   row_charges = fuse_charges([indices[n].charges for n in range(rank // 2)],
#                              [1 for _ in range(rank // 2)])
#   column_charges = fuse_charges(
#       [indices[n].charges for n in range(rank // 2, rank)],
#       [1 for _ in range(rank // 2, rank)])

#   i01 = indices[0] * indices[1]
#   i23 = indices[2] * indices[3]
#   positions = find_dense_positions(i01.charges, 1, i23.charges, 1, 0)
#   assert len(positions) == n1

# def test_find_sparse_positions():
#   D = 40  #bond dimension
#   B = 4  #number of blocks
#   dtype = np.int16  #the dtype of the quantum numbers
#   rank = 4
#   flows = np.asarray([1 for _ in range(rank)])
#   flows[-2::] = -1
#   charges = [
#       np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#       for _ in range(rank)
#   ]
#   indices = [
#       Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
#       for n in range(rank)
#   ]
#   n1 = compute_num_nonzero([i.charges for i in indices],
#                            [i.flow for i in indices])
#   row_charges = fuse_charges([indices[n].charges for n in range(rank // 2)],
#                              [1 for _ in range(rank // 2)])
#   column_charges = fuse_charges(
#       [indices[n].charges for n in range(rank // 2, rank)],
#       [1 for _ in range(rank // 2, rank)])

#   i01 = indices[0] * indices[1]
#   i23 = indices[2] * indices[3]
#   unique_row_charges = np.unique(i01.charges)
#   unique_column_charges = np.unique(i23.charges)
#   common_charges = np.intersect1d(
#       unique_row_charges, -unique_column_charges, assume_unique=True)
#   blocks = find_sparse_positions(
#       i01.charges, 1, i23.charges, 1, target_charges=[0])
#   assert sum([len(v) for v in blocks.values()]) == n1
#   np.testing.assert_allclose(np.sort(blocks[0]), np.arange(n1))

# def test_find_sparse_positions_2():
#   D = 40  #bond dimension
#   B = 4  #number of blocks
#   dtype = np.int16  #the dtype of the quantum numbers
#   flows = [1, -1]

#   rank = len(flows)
#   charges = [
#       np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#       for _ in range(rank)
#   ]
#   indices = [
#       Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
#       for n in range(rank)
#   ]
#   i1, i2 = indices
#   common_charges = np.intersect1d(i1.charges, i2.charges)
#   row_locations = find_sparse_positions(
#       left_charges=i1.charges,
#       left_flow=flows[0],
#       right_charges=i2.charges,
#       right_flow=flows[1],
#       target_charges=common_charges)
#   fused = (i1 * i2).charges
#   relevant = fused[np.isin(fused, common_charges)]
#   for k, v in row_locations.items():
#     np.testing.assert_allclose(np.nonzero(relevant == k)[0], np.sort(v))

# def test_get_diagonal_blocks():
#   D = 40  #bond dimension
#   B = 4  #number of blocks
#   dtype = np.int16  #the dtype of the quantum numbers
#   rank = 4
#   flows = np.asarray([1 for _ in range(rank)])
#   flows[-2::] = -1
#   charges = [
#       np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#       for _ in range(rank)
#   ]
#   indices = [
#       Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
#       for n in range(rank)
#   ]
#   common_charges = np.intersect1d(indices[0].charges, indices[1].charges)
#   row_locations = find_sparse_positions(
#       left_charges=indices[0].charges,
#       left_flow=1,
#       right_charges=indices[1].charges,
#       right_flow=1,
#       target_charges=common_charges)

# def test_dense_transpose():
#   Ds = [10, 11, 12]  #bond dimension
#   rank = len(Ds)
#   flows = np.asarray([1 for _ in range(rank)])
#   flows[-2::] = -1
#   charges = [np.zeros(Ds[n], dtype=np.int16) for n in range(rank)]
#   indices = [
#       Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
#       for n in range(rank)
#   ]
#   A = BlockSparseTensor.random(indices=indices, dtype=np.float64)
#   B = np.transpose(np.reshape(A.data.copy(), Ds), (1, 0, 2))
#   A.transpose((1, 0, 2))
#   np.testing.assert_allclose(A.data, B.flat)
