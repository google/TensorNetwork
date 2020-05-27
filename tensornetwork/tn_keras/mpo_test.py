import itertools
from tensornetwork.tn_keras import mpo
import tensorflow as tf
import numpy as np
import pytest


@pytest.mark.parametrize('in_dim_base,dim1,dim2,num_nodes,bond_dim',
    itertools.product(
      [3, 4],
      [3, 4], 
      [2, 5], 
      [3, 4],
      [2, 3])
)
def test_shape_sanity_check(in_dim_base,dim1, dim2, num_nodes, bond_dim):
  model = tf.keras.Sequential([
    tf.keras.Input(in_dim_base**num_nodes),
    mpo.DenseMPO(dim1**num_nodes, num_nodes=num_nodes, bond_dim=bond_dim), 
    mpo.DenseMPO(dim2**num_nodes, num_nodes=num_nodes, bond_dim=bond_dim),
  ])
  
  model.predict(np.ones((32, in_dim_base**num_nodes)))