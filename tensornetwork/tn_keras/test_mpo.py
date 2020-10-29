import itertools
from tensornetwork.tn_keras.layers import DenseMPO
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
import numpy as np
import pytest


@pytest.mark.parametrize('in_dim_base,dim1,dim2,num_nodes,bond_dim',
                         itertools.product([3, 4], [3, 4], [2, 5], [3, 4],
                                           [2, 3]))
def test_shape_sanity_check(in_dim_base, dim1, dim2, num_nodes, bond_dim):
  model = Sequential([
      Input(in_dim_base**num_nodes),
      DenseMPO(dim1**num_nodes, num_nodes=num_nodes, bond_dim=bond_dim),
      DenseMPO(dim2**num_nodes, num_nodes=num_nodes, bond_dim=bond_dim),
  ])
  # Hard code batch size.
  result = model.predict(np.ones((32, in_dim_base**num_nodes)))
  assert result.shape == (32, dim2**num_nodes)
