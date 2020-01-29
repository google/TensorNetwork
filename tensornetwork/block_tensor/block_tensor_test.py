import numpy as np
import pytest

from tensornetwork.block_tensor.charge import U1Charge, fuse_charges
from tensornetwork.block_tensor.index import Index
from tensornetwork.block_tensor.block_tensor import compute_num_nonzero, reduce_charges, BlockSparseTensor, fuse_ndarrays

np_dtypes = [np.float32, np.float16, np.float64, np.complex64, np.complex128]


@pytest.mark.parametrize("dtype", np_dtypes)
def test_block_sparse_init(dtype):
  D = 10  #bond dimension
  B = 10  #number of blocks
  rank = 4
  flows = np.asarray([False for _ in range(rank)])
  flows[-2::] = True
  charges = [
      U1Charge(np.random.randint(-B // 2, B // 2 + 1, D).astype(np.int16))
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


def test_find_dense_positions():
  left_charges = np.asarray([-2, 0, 1, 0, 0]).astype(np.int16)
  right_charges = np.asarray([-1, 0, 2, 1]).astype(np.int16)
  target_charge = np.zeros((1, 1), dtype=np.int16)
  fused_charges = fuse_ndarrays([left_charges, right_charges])
  dense_positions = reduce_charges(
      [U1Charge(left_charges), U1Charge(right_charges)], [False, False],
      target_charge,
      return_locations=True)
  np.testing.assert_allclose(
      dense_positions[1],
      np.nonzero(fused_charges == target_charge[0, 0])[0])


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
#       Index(
#           charges=U1Charge(charges[n]), flow=flows[n], name='index{}'.format(n))
#       for n in range(rank)
#   ]
#   n1 = compute_num_nonzero([i.charges for i in indices],
#                            [i.flow for i in indices])

#   i01 = indices[0] * indices[1]
#   i23 = indices[2] * indices[3]
#   positions = find_dense_positions([i01.charges, i23.charges], [1, 1],
#                                    U1Charge(np.asarray([0])))
#   assert len(positions[0]) == n1

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
#       Index(
#           charges=U1Charge(charges[n]), flow=flows[n], name='index{}'.format(n))
#       for n in range(rank)
#   ]
#   n1 = compute_num_nonzero([i.charges for i in indices],
#                            [i.flow for i in indices])
#   i01 = indices[0] * indices[1]
#   i23 = indices[2] * indices[3]
#   unique_row_charges = np.unique(i01.charges.charges)
#   unique_column_charges = np.unique(i23.charges.charges)
#   common_charges = np.intersect1d(
#       unique_row_charges, -unique_column_charges, assume_unique=True)
#   blocks = find_sparse_positions([i01.charges, i23.charges], [1, 1],
#                                  target_charges=U1Charge(np.asarray([0])))
#   assert sum([len(v) for v in blocks.values()]) == n1
#   np.testing.assert_allclose(np.sort(blocks[0]), np.arange(n1))

# def test_find_sparse_positions_2():
#   D = 1000  #bond dimension
#   B = 4  #number of blocks
#   dtype = np.int16  #the dtype of the quantum numbers
#   charges = np.random.randint(-B // 2, B // 2 + 1, D).astype(dtype)
#   index = Index(charges=U1Charge(charges), flow=1, name='index0')
#   targets = np.asarray([-1, 0, 1])
#   blocks = find_sparse_positions([index.charges], [index.flow],
#                                  target_charges=U1Charge(targets))

#   inds = np.isin(charges, targets)
#   relevant_charges = charges[inds]
#   blocks_ = {t: np.nonzero(relevant_charges == t)[0] for t in targets}
#   assert np.all(
#       np.asarray(list(blocks.keys())) == np.asarray(list(blocks_.keys())))
#   for k in blocks.keys():
#     assert np.all(blocks[k] == blocks_[k])

# def test_find_sparse_positions_3():
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
#       Index(
#           charges=U1Charge(charges[n]), flow=flows[n], name='index{}'.format(n))
#       for n in range(rank)
#   ]
#   i1, i2 = indices
#   common_charges = np.intersect1d(i1.charges.charges, i2.charges.charges)
#   row_locations = find_sparse_positions(
#       charges=[i1.charges, i2.charges],
#       flows=flows,
#       target_charges=U1Charge(common_charges))
#   fused = (i1 * i2).charges
#   relevant = fused.charges[np.isin(fused.charges, common_charges)]
#   for k, v in row_locations.items():
#     np.testing.assert_allclose(np.nonzero(relevant == k)[0], np.sort(v))

# # def test_dense_transpose():
# #   Ds = [10, 11, 12]  #bond dimension
# #   rank = len(Ds)
# #   flows = np.asarray([1 for _ in range(rank)])
# #   flows[-2::] = -1
# #   charges = [U1Charge(np.zeros(Ds[n], dtype=np.int16)) for n in range(rank)]
# #   indices = [
# #       Index(charges=charges[n], flow=flows[n], name='index{}'.format(n))
# #       for n in range(rank)
# #   ]
# #   A = BlockSparseTensor.random(indices=indices, dtype=np.float64)
# #   B = np.transpose(np.reshape(A.data.copy(), Ds), (1, 0, 2))
# #   A.transpose((1, 0, 2))
# #   np.testing.assert_allclose(A.data, B.flat)

# #   B = np.transpose(np.reshape(A.data.copy(), [11, 10, 12]), (1, 0, 2))
# #   A.transpose((1, 0, 2))

# #   np.testing.assert_allclose(A.data, B.flat)

# @pytest.mark.parametrize("R", [1, 2])
# def test_find_diagonal_dense_blocks(R):
#   rs = [U1Charge(np.random.randint(-4, 4, 50)) for _ in range(R)]
#   cs = [U1Charge(np.random.randint(-4, 4, 50)) for _ in range(R)]
#   charges = rs + cs

#   left_fused = fuse_charges(charges[0:R], [1] * R)
#   right_fused = fuse_charges(charges[R:], [1] * R)
#   left_unique = left_fused.unique()
#   right_unique = right_fused.unique()
#   zero = left_unique.zero_charge
#   blocks = {}
#   rdim = len(right_fused)
#   for lu in left_unique:
#     linds = np.nonzero(left_fused == lu)[0]
#     rinds = np.nonzero(right_fused == lu * (-1))[0]
#     if (len(linds) > 0) and (len(rinds) > 0):
#       blocks[lu] = fuse_ndarrays([linds * rdim, rinds])
#   comm, blocks_ = _find_diagonal_dense_blocks(rs, cs, [1] * R, [1] * R)
#   for n in range(len(comm)):
#     assert np.all(blocks[comm.charges[n]] == blocks_[n][0])

# # #@pytest.mark.parametrize("dtype", np_dtypes)
# # def test_find_diagonal_dense_blocks_2():
# #   R = 1
# #   rs = [U1Charge(np.random.randint(-4, 4, 50)) for _ in range(R)]
# #   cs = [U1Charge(np.random.randint(-4, 4, 50)) for _ in range(R)]
# #   charges = rs + cs

# #   left_fused = fuse_charges(charges[0:R], [1] * R)
# #   right_fused = fuse_charges(charges[R:], [1] * R)
# #   left_unique = left_fused.unique()
# #   right_unique = right_fused.unique()
# #   zero = left_unique.zero_charge
# #   blocks = {}
# #   rdim = len(right_fused)
# #   for lu in left_unique:
# #     linds = np.nonzero(left_fused == lu)[0]
# #     rinds = np.nonzero(right_fused == lu * (-1))[0]
# #     if (len(linds) > 0) and (len(rinds) > 0):
# #       blocks[lu] = fuse_ndarrays([linds * rdim, rinds])
# #   comm, blocks_ = _find_diagonal_dense_blocks(rs, cs, [1] * R, [1] * R)
# #   for n in range(len(comm)):
# #     assert np.all(blocks[comm.charges[n]] == blocks_[n][0])

# @pytest.mark.parametrize("R", [1, 2])
# def test_find_diagonal_dense_blocks_transposed(R):
#   order = np.arange(2 * R)
#   np.random.shuffle(order)
#   rs = [U1Charge(np.random.randint(-4, 4, 50)) for _ in range(R)]
#   cs = [U1Charge(np.random.randint(-4, 4, 40)) for _ in range(R)]
#   charges = rs + cs
#   dims = np.asarray([len(c) for c in charges])
#   strides = np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))
#   stride_arrays = [np.arange(dims[n]) * strides[n] for n in range(2 * R)]

#   left_fused = fuse_charges([charges[n] for n in order[0:R]], [1] * R)
#   right_fused = fuse_charges([charges[n] for n in order[R:]], [1] * R)
#   lstrides = fuse_ndarrays([stride_arrays[n] for n in order[0:R]])
#   rstrides = fuse_ndarrays([stride_arrays[n] for n in order[R:]])

#   left_unique = left_fused.unique()
#   right_unique = right_fused.unique()
#   blocks = {}
#   rdim = len(right_fused)
#   for lu in left_unique:
#     linds = np.nonzero(left_fused == lu)[0]
#     rinds = np.nonzero(right_fused == lu * (-1))[0]
#     if (len(linds) > 0) and (len(rinds) > 0):
#       tmp = fuse_ndarrays([linds * rdim, rinds])
#       blocks[lu] = _find_values_in_fused(tmp, lstrides, rstrides)

#   comm, blocks_ = _find_diagonal_dense_blocks([charges[n] for n in order[0:R]],
#                                               [charges[n] for n in order[R:]],
#                                               [1] * R, [1] * R,
#                                               row_strides=strides[order[0:R]],
#                                               column_strides=strides[order[R:]])
#   for n in range(len(comm)):
#     assert np.all(blocks[comm.charges[n]] == blocks_[n][0])
