import numpy as np
import pytest
# pylint: disable=line-too-long
from tensornetwork.block_tensor.block_tensor import BlockSparseTensor, compute_num_nonzero
from index import Index

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
