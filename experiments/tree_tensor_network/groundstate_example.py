# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tree Tensor Network for the groundstate of the Transverse Ising chain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from experiments.tree_tensor_network import ttn_1d_uniform

if __name__ == "__main__":
  backend = "tensorflow"  # "numpy" and "jax" are also supported!

  if backend == "tensorflow":
    import tensorflow as tf
    tf.enable_v2_behavior()
    dtype = tf.float64
  elif backend == "jax":
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as np
    dtype = np.float64
  elif backend == "numpy":
    import numpy as np
    dtype = np.float64

  ttn_1d_uniform.set_backend(backend)

  num_layers = 6
  max_bond_dim = 16
  build_graphs = True

  num_sweeps = 1000

  Ds = [min(2**i, max_bond_dim) for i in range(1, num_layers + 1)]

  print("----------------------------------------------------")
  print("Variational ground state optimization.")
  print("----------------------------------------------------")
  print("System size:", 2**num_layers)
  print("Bond dimensions:", Ds)

  H = ttn_1d_uniform.get_ham_ising(dtype)
  isos_012 = ttn_1d_uniform.random_tree_tn_uniform(Ds, dtype, top_rank=1)

  isos_012 = ttn_1d_uniform.opt_tree_energy(
      isos_012,
      H,
      num_sweeps,
      1,
      verbose=1,
      graphed=build_graphs,
      ham_shift=0.2)
