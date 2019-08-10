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
superoperators, as well as environment tensors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import jax
import jax.config
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import tensornetwork
from tensornetwork.contractors.opt_einsum_paths import path_contractors


@jax.jit
def binary_mera_energy(hamiltonian, state, isometry, disentangler):
  """Computes the energy using a layer of uniform binary MERA.

  Args:
    hamiltonian: The hamiltonian defined at the bottom of the MERA layer.
    state: The 3-site reduced state defined at the top of the MERA layer.
    isometry: The isometry tensor of the binary MERA.
    disentangler: The disentangler tensor of the binary MERA.

  Returns:
    The energy.
  """
  out = []
  for dirn in ('left', 'right'):
    net = tensornetwork.TensorNetwork(backend="jax")

    iso_l = net.add_node(isometry)
    iso_c = net.add_node(isometry)
    iso_r = net.add_node(isometry)

    iso_l_con = net.add_node(np.conj(isometry))
    iso_c_con = net.add_node(np.conj(isometry))
    iso_r_con = net.add_node(np.conj(isometry))

    op = net.add_node(hamiltonian)
    rho = net.add_node(state)

    un_l = net.add_node(disentangler)
    un_l_con = net.add_node(np.conj(disentangler))

    un_r = net.add_node(disentangler)
    un_r_con = net.add_node(np.conj(disentangler))

    net.connect(iso_l[2], rho[0])
    net.connect(iso_c[2], rho[1])
    net.connect(iso_r[2], rho[2])

    net.connect(iso_l[0], iso_l_con[0])
    net.connect(iso_l[1], un_l[2])
    net.connect(iso_c[0], un_l[3])
    net.connect(iso_c[1], un_r[2])
    net.connect(iso_r[0], un_r[3])
    net.connect(iso_r[1], iso_r_con[1])

    if dirn == 'right':
      net.connect(un_l[0], un_l_con[0])
      net.connect(un_l[1], op[3])
      net.connect(un_r[0], op[4])
      net.connect(un_r[1], op[5])
      net.connect(op[0], un_l_con[1])
      net.connect(op[1], un_r_con[0])
      net.connect(op[2], un_r_con[1])
    elif dirn == 'left':
      net.connect(un_l[0], op[3])
      net.connect(un_l[1], op[4])
      net.connect(un_r[0], op[5])
      net.connect(un_r[1], un_r_con[1])
      net.connect(op[0], un_l_con[0])
      net.connect(op[1], un_l_con[1])
      net.connect(op[2], un_r_con[0])

    net.connect(un_l_con[2], iso_l_con[1])
    net.connect(un_l_con[3], iso_c_con[0])
    net.connect(un_r_con[2], iso_c_con[1])
    net.connect(un_r_con[3], iso_r_con[0])

    net.connect(iso_l_con[2], rho[3])
    net.connect(iso_c_con[2], rho[4])
    net.connect(iso_r_con[2], rho[5])

    out.append(path_contractors.auto(net).get_final_node().get_tensor())

  return 0.5 * sum(out)


"""Descending super-operator."""
descend = jax.jit(jax.grad(binary_mera_energy, argnums=0, holomorphic=True))


"""Ascending super-operator."""
ascend = jax.jit(jax.grad(binary_mera_energy, argnums=1, holomorphic=True))


"""Isometry environment."""
env_iso = jax.jit(jax.grad(binary_mera_energy, argnums=2, holomorphic=False))


"""Disentangler environment."""
env_dis = jax.jit(jax.grad(binary_mera_energy, argnums=3, holomorphic=False))


@jax.jit
def update_iso(hamiltonian, state, isometry, disentangler):
  """Updates the isometry with the aim of reducing the energy."""
  env = env_iso(hamiltonian, state, isometry, disentangler)

  net = tensornetwork.TensorNetwork(backend="jax")
  nenv = net.add_node(env, axis_names=["l", "r", "t"])
  output_edges = [nenv["l"], nenv["r"], nenv["t"]]

  nu, ns, nv, _ = net.split_node_full_svd(
    nenv, [nenv["l"], nenv["r"]], [nenv["t"]])
  _, s_edges = net.remove_node(ns)
  net.connect(s_edges[0], s_edges[1])
  nres = net.contract_between(nu, nv, output_edge_order=output_edges)

  return np.conj(nres.get_tensor())


@jax.jit
def update_dis(hamiltonian, state, isometry, disentangler):
  """Updates the disentangler with the aim of reducing the energy."""
  env = env_dis(hamiltonian, state, isometry, disentangler)

  net = tensornetwork.TensorNetwork(backend="jax")
  nenv = net.add_node(env, axis_names=["bl", "br", "tl", "tr"])
  output_edges = [nenv["bl"], nenv["br"], nenv["tl"], nenv["tr"]]

  nu, ns, nv, _ = net.split_node_full_svd(
    nenv, [nenv["bl"], nenv["br"]], [nenv["tl"], nenv["tr"]])
  _, s_edges = net.remove_node(ns)
  net.connect(s_edges[0], s_edges[1])
  nres = net.contract_between(nu, nv, output_edge_order=output_edges)

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
  return np.reshape(hmat, [2]*6)


def optimize_linear(hamiltonian, state, isometry, disentangler, num_itr):
  """Optimize a scale-invariant MERA using linearized updates.
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
  """
  E = np.array([[1, 0], [0, 1]])
  X = np.array([[0, 1], [1, 0]])
  Z = np.array([[1, 0], [0, -1]])
  hmat = np.kron(X, np.kron(Z, X))
  hmat -= 0.5 * (np.kron(np.kron(X, X), E) + np.kron(E, np.kron(X, X)))
  return np.reshape(hmat, [2]*6)


if __name__ == '__main__':
  h = ham_ising()
  s = np.reshape(np.eye(2**3), [2]*6) / 2**3
  dis = np.reshape(np.eye(2**2), [2]*4)
  iso = disentangler[:,:,:,0]

  optimize_linear(ham, s, iso, dis, 100)