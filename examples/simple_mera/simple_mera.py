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
"""Very simple scale-invariant MERA.

Uses automatic differentiation to construct ascending and descending
superoperators, as well as environment tensors so that only a single tensor
network needs to be defined: The network for computing the energy using a
single layer of MERA together with a reduced state and a hamiltonian.

Run as the main module, this module executes a script that optimizes a MERA
for the critical Ising model.

For the scale-invariant MERA, see arXiv:1109.5334.
"""

import jax
import jax.config
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import tensornetwork
import tensornetwork.linalg.node_linalg
from tensornetwork import contractors


@jax.jit
def binary_mera_energy(hamiltonian, state, isometry, disentangler):
  """Computes the energy using a layer of uniform binary MERA.

  Args:
    hamiltonian: The hamiltonian (rank-6 tensor) defined at the bottom of the
      MERA layer.
    state: The 3-site reduced state (rank-6 tensor) defined at the top of the
      MERA layer.
    isometry: The isometry tensor (rank 3) of the binary MERA.
    disentangler: The disentangler tensor (rank 4) of the binary MERA.

  Returns:
    The energy.
  """
  backend = "jax"

  out = []
  for dirn in ('left', 'right'):
    iso_l = tensornetwork.Node(isometry, backend=backend)
    iso_c = tensornetwork.Node(isometry, backend=backend)
    iso_r = tensornetwork.Node(isometry, backend=backend)

    iso_l_con = tensornetwork.linalg.node_linalg.conj(iso_l)
    iso_c_con = tensornetwork.linalg.node_linalg.conj(iso_c)
    iso_r_con = tensornetwork.linalg.node_linalg.conj(iso_r)

    op = tensornetwork.Node(hamiltonian, backend=backend)
    rho = tensornetwork.Node(state, backend=backend)

    un_l = tensornetwork.Node(disentangler, backend=backend)
    un_l_con = tensornetwork.linalg.node_linalg.conj(un_l)

    un_r = tensornetwork.Node(disentangler, backend=backend)
    un_r_con = tensornetwork.linalg.node_linalg.conj(un_r)

    tensornetwork.connect(iso_l[2], rho[0])
    tensornetwork.connect(iso_c[2], rho[1])
    tensornetwork.connect(iso_r[2], rho[2])

    tensornetwork.connect(iso_l[0], iso_l_con[0])
    tensornetwork.connect(iso_l[1], un_l[2])
    tensornetwork.connect(iso_c[0], un_l[3])
    tensornetwork.connect(iso_c[1], un_r[2])
    tensornetwork.connect(iso_r[0], un_r[3])
    tensornetwork.connect(iso_r[1], iso_r_con[1])

    if dirn == 'right':
      tensornetwork.connect(un_l[0], un_l_con[0])
      tensornetwork.connect(un_l[1], op[3])
      tensornetwork.connect(un_r[0], op[4])
      tensornetwork.connect(un_r[1], op[5])
      tensornetwork.connect(op[0], un_l_con[1])
      tensornetwork.connect(op[1], un_r_con[0])
      tensornetwork.connect(op[2], un_r_con[1])
    elif dirn == 'left':
      tensornetwork.connect(un_l[0], op[3])
      tensornetwork.connect(un_l[1], op[4])
      tensornetwork.connect(un_r[0], op[5])
      tensornetwork.connect(un_r[1], un_r_con[1])
      tensornetwork.connect(op[0], un_l_con[0])
      tensornetwork.connect(op[1], un_l_con[1])
      tensornetwork.connect(op[2], un_r_con[0])

    tensornetwork.connect(un_l_con[2], iso_l_con[1])
    tensornetwork.connect(un_l_con[3], iso_c_con[0])
    tensornetwork.connect(un_r_con[2], iso_c_con[1])
    tensornetwork.connect(un_r_con[3], iso_r_con[0])

    tensornetwork.connect(iso_l_con[2], rho[3])
    tensornetwork.connect(iso_c_con[2], rho[4])
    tensornetwork.connect(iso_r_con[2], rho[5])

    # FIXME: Check that this is giving us a good path!
    out.append(
        contractors.branch(tensornetwork.reachable(rho),
                           nbranch=2).get_tensor())

  return 0.5 * sum(out)


descend = jax.jit(jax.grad(binary_mera_energy, argnums=0, holomorphic=True))
"""Descending super-operator.

Args:
  hamiltonian: A dummy rank-6 tensor not involved in the computation.
  state: The 3-site reduced state to be descended (rank-6 tensor).
  isometry: The isometry tensor of the binary MERA.
  disentangler: The disentangler tensor of the binary MERA.

Returns:
  The descended state (spatially averaged).
"""

ascend = jax.jit(jax.grad(binary_mera_energy, argnums=1, holomorphic=True))
"""Ascending super-operator.

Args:
  operator: The operator to be ascended (rank-6 tensor).
  state: A dummy rank-6 tensor not involved in the computation.
  isometry: The isometry tensor of the binary MERA.
  disentangler: The disentangler tensor of the binary MERA.

Returns:
  The ascended operator (spatially averaged).
"""

# NOTE: Not a holomorphic function, but a real-valued loss function.
env_iso = jax.jit(jax.grad(binary_mera_energy, argnums=2, holomorphic=True))
"""Isometry environment tensor.

In other words: The derivative of the `binary_mera_energy()` with respect to
the isometry tensor.

Args:
  hamiltonian: The hamiltonian (rank-6 tensor) defined at the bottom of the
    MERA layer.
  state: The 3-site reduced state (rank-6 tensor) defined at the top of the
    MERA layer.
  isometry: A dummy isometry tensor (rank 3) not used in the computation.
  disentangler: The disentangler tensor (rank 4) of the binary MERA.

Returns:
  The environment tensor of the isometry, including all contributions.
"""

# NOTE: Not a holomorphic function, but a real-valued loss function.
env_dis = jax.jit(jax.grad(binary_mera_energy, argnums=3, holomorphic=True))
"""Disentangler environment.

In other words: The derivative of the `binary_mera_energy()` with respect to
the disentangler tensor.

Args:
  hamiltonian: The hamiltonian (rank-6 tensor) defined at the bottom of the
    MERA layer.
  state: The 3-site reduced state (rank-6 tensor) defined at the top of the
    MERA layer.
  isometry: The isometry tensor (rank 3) of the binary MERA.
  disentangler: A dummy disentangler (rank 4) not used in the computation.

Returns:
  The environment tensor of the disentangler, including all contributions.
"""


@jax.jit
def update_iso(hamiltonian, state, isometry, disentangler):
  """Updates the isometry with the aim of reducing the energy.

  Args:
    hamiltonian: The hamiltonian (rank-6 tensor) defined at the bottom of the
      MERA layer.
    state: The 3-site reduced state (rank-6 tensor) defined at the top of the
      MERA layer.
    isometry: The isometry tensor (rank 3) of the binary MERA.
    disentangler: The disentangler tensor (rank 4) of the binary MERA.

  Returns:
    The updated isometry.
  """
  env = env_iso(hamiltonian, state, isometry, disentangler)

  nenv = tensornetwork.Node(env, axis_names=["l", "r", "t"], backend="jax")
  output_edges = [nenv["l"], nenv["r"], nenv["t"]]

  nu, _, nv, _ = tensornetwork.split_node_full_svd(
      nenv, [nenv["l"], nenv["r"]], [nenv["t"]],
      left_edge_name="s1",
      right_edge_name="s2")
  nu["s1"].disconnect()
  nv["s2"].disconnect()
  tensornetwork.connect(nu["s1"], nv["s2"])
  nres = tensornetwork.contract_between(nu, nv, output_edge_order=output_edges)

  return np.conj(nres.get_tensor())


@jax.jit
def update_dis(hamiltonian, state, isometry, disentangler):
  """Updates the disentangler with the aim of reducing the energy.

  Args:
    hamiltonian: The hamiltonian (rank-6 tensor) defined at the bottom of the
      MERA layer.
    state: The 3-site reduced state (rank-6 tensor) defined at the top of the
      MERA layer.
    isometry: The isometry tensor (rank 3) of the binary MERA.
    disentangler: The disentangler tensor (rank 4) of the binary MERA.

  Returns:
    The updated disentangler.
  """
  env = env_dis(hamiltonian, state, isometry, disentangler)

  nenv = tensornetwork.Node(
      env, axis_names=["bl", "br", "tl", "tr"], backend="jax")
  output_edges = [nenv["bl"], nenv["br"], nenv["tl"], nenv["tr"]]

  nu, _, nv, _ = tensornetwork.split_node_full_svd(
      nenv, [nenv["bl"], nenv["br"]], [nenv["tl"], nenv["tr"]],
      left_edge_name="s1",
      right_edge_name="s2")
  nu["s1"].disconnect()
  nv["s2"].disconnect()
  tensornetwork.connect(nu["s1"], nv["s2"])
  nres = tensornetwork.contract_between(nu, nv, output_edge_order=output_edges)

  return np.conj(nres.get_tensor())


def shift_ham(hamiltonian, shift=None):
  """Applies a shift to a hamiltonian.

  Args:
    hamiltonian: The hamiltonian tensor (rank 6).
    shift: The amount by which to shift. If `None`, shifts so that the local
      term is negative semi-definite.

  Returns:
    The shifted Hamiltonian.
  """
  hmat = np.reshape(hamiltonian, (2**3, -1))
  if shift is None:
    shift = np.amax(np.linalg.eigh(hmat)[0])
  hmat -= shift * np.eye(2**3)
  return np.reshape(hmat, [2] * 6)


def optimize_linear(hamiltonian, state, isometry, disentangler, num_itr):
  """Optimize a scale-invariant MERA using linearized updates.

  The MERA is assumed to be completely uniform and scale-invariant, consisting
  of a single isometry and disentangler.

  Args:
    hamiltonian: The hamiltonian (rank-6 tensor) defined at the bottom.
    state: An initial 3-site reduced state (rank-6 tensor) to initialize the
      descending fixed-point computation.
    isometry: The isometry tensor (rank 3) of the binary MERA.
    disentangler: The disentangler tensor (rank 4) of the binary MERA.

  Returns:
    state: The approximate descending fixed-point reduced state (rank 6).
    isometry: The optimized isometry.
    disentangler: The optimized disentangler.
  """
  h_shifted = shift_ham(hamiltonian)

  for i in range(num_itr):
    isometry = update_iso(h_shifted, state, isometry, disentangler)
    disentangler = update_dis(h_shifted, state, isometry, disentangler)

    for _ in range(10):
      state = descend(hamiltonian, state, isometry, disentangler)

    en = binary_mera_energy(hamiltonian, state, isometry, disentangler)
    print("{}:\t{}".format(i, en))

  return state, isometry, disentangler


def ham_ising():
  """Dimension 2 "Ising" Hamiltonian.

  This version from Evenbly & White, Phys. Rev. Lett. 116, 140403
  (2016).
  """
  E = np.array([[1, 0], [0, 1]])
  X = np.array([[0, 1], [1, 0]])
  Z = np.array([[1, 0], [0, -1]])
  hmat = np.kron(X, np.kron(Z, X))
  hmat -= 0.5 * (np.kron(np.kron(X, X), E) + np.kron(E, np.kron(X, X)))
  return np.reshape(hmat, [2] * 6)


if __name__ == '__main__':
  # Starting from a very simple initial MERA, optimize for the critical Ising
  # model.
  h = ham_ising()
  s = np.reshape(np.eye(2**3), [2] * 6) / 2**3
  dis = np.reshape(np.eye(2**2), [2] * 4)
  iso = dis[:, :, :, 0]

  s, iso, dis = optimize_linear(h, s, iso, dis, 100)
