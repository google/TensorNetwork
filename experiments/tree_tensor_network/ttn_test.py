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
"""
Tree tensor network unit tests.
"""
import pytest
import numpy as np
import tensorflow as tf
from jax.config import config
config.update("jax_enable_x64", True)
import tensornetwork as tn
import examples.wavefunctions.wavefunctions as wf
import experiments.tree_tensor_network.ttn_1d_uniform as ttn

tf.enable_v2_behavior()


def test_opt(backend):
  if backend == "tensorflow":
    dtype = tf.float64
  else:
    dtype = np.float64

  num_layers = 3
  max_bond_dim = 8
  build_graphs = True
  num_sweeps = 5

  Ds = [min(2**i, max_bond_dim) for i in range(1, num_layers + 1)]

  H = ttn.get_ham_ising(dtype)
  isos_012 = ttn.random_tree_tn_uniform(Ds, dtype, top_rank=1)
  energy_0 = ttn.backend.trace(ttn.top_hamiltonian(H, isos_012))
  isos_012 = ttn.opt_tree_energy(
    isos_012,
    H,
    num_sweeps,
    1,
    verbose=0,
    graphed=build_graphs,
    ham_shift=0.2)
  energy_1 = ttn.backend.trace(ttn.top_hamiltonian(H, isos_012))
  assert ttn.backend.to_numpy(energy_1) < ttn.backend.to_numpy(energy_0)

  N = 2**num_layers
  full_state = ttn.descend_full_state_pure(isos_012)
  norm = ttn.backend.norm(ttn.backend.reshape(full_state, (2**N,)))
  assert abs(ttn.backend.to_numpy(norm) - 1) < 1e-12

  if backend != "jax":
    # wavefunctions assumes TensorFlow. This will interact with numpy OK, but
    # not JAX.
    h = ttn._dense_ham_term(H)
    energy_1_full_state = sum(
      wf.expval(full_state, h, j, pbc=True) for j in range(N))
    assert abs(ttn.backend.to_numpy(energy_1_full_state) -
               ttn.backend.to_numpy(energy_1)) < 1e-12

  isos_012 = ttn.opt_tree_energy(
    isos_012,
    H,
    1,
    1,
    verbose=0,
    graphed=False,
    decomp_mode="eigh",
    ham_shift=0.2)

  for iso in isos_012:
    assert ttn.backend.to_numpy(ttn.check_iso(iso)) < 1e-6

  isos_012 = ttn.opt_tree_energy(
    isos_012,
    H,
    1,
    1,
    verbose=0,
    graphed=False,
    decomp_mode="svd",
    ham_shift=0.2)

  for iso in isos_012:
    assert ttn.backend.to_numpy(ttn.check_iso(iso)) < 1e-6

  isos_012 = ttn.opt_tree_energy(
    isos_012,
    H,
    1,
    1,
    verbose=0,
    graphed=False,
    decomp_mode="svd_full_iso_scipy",
    ham_shift=0.2)

  for iso in isos_012:
    assert ttn.backend.to_numpy(ttn.check_iso(iso)) < 1e-12


def test_expvals(random_isos):
  H = ttn.get_ham_ising(random_isos[0].dtype)
  ens = ttn.backend.to_numpy(ttn.tree_energy_expval_check(H, random_isos))
  assert np.all(ens - ens[0] < 1e-12)


def test_iso(random_isos):
  for iso in random_isos:
    assert ttn.backend.to_numpy(ttn.check_iso(iso)) < 1e-12


@pytest.fixture(params=["tensorflow", "jax", "numpy"])
def backend(request):
  backend_name = request.param
  ttn.set_backend(backend_name)
  return backend_name


@pytest.fixture(
    params=[
        ("tensorflow", 3, 4, 2, tf.complex128),
        ("numpy", 1, 4, 2, np.complex128),
        ("numpy", 2, 8, 1, np.float64),
        ("numpy", 5, 3, 3, np.float64)]
    )
def random_isos(request):
  backend, num_layers, max_bond_dim, top_rank, dtype = request.param
  ttn.set_backend(backend)
  Ds = [min(2**i, max_bond_dim) for i in range(1, num_layers + 1)]
  return ttn.random_tree_tn_uniform(Ds, dtype, top_rank=top_rank)