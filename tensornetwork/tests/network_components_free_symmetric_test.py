import numpy as np
import pytest
import tensornetwork as tn
from tensornetwork.block_sparse import (BlockSparseTensor, Index, BaseCharge,
                                        U1Charge)


@pytest.mark.parametrize("num_charges", [1, 2])
def test_sparse_shape(num_charges):
  np.random.seed(10)
  dtype = np.float64
  shape = [10, 11, 12, 13]
  R = len(shape)
  charges = [
      BaseCharge(
          np.random.randint(-5, 5, (shape[n], num_charges)),
          charge_types=[U1Charge] * num_charges) for n in range(R)
  ]
  flows = list(np.full(R, fill_value=False, dtype=np.bool))
  indices = [Index(charges[n], flows[n]) for n in range(R)]
  a = BlockSparseTensor.random(indices=indices, dtype=dtype)
  node = tn.Node(a, backend='symmetric')
  for s1, s2 in zip(node.sparse_shape, indices):
    assert s1 == s2
