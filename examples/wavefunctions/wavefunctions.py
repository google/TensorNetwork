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
"""Trotter evolution of exact wavefunctions."""

from functools import reduce
import sys
import tensorflow as tf

import tensornetwork
from examples.wavefunctions.trotter import trotter_prepare_gates


def inner(psi1, psi2):
  """Computes the inner product <psi1|psi2>.

  Args:
    psi1: A tensor representing the first wavefunction.
    psi2: A tensor representing the second wavefunction.

  Returns:
    inner_product: The vector inner product.
  """
  return tf.reduce_sum(tf.math.conj(psi1) * psi2)


def apply_op(psi, op, n1, pbc=False):
  """Apply a local operator to a wavefunction.

  The number of dimensions of the tensor representing the wavefunction `psi`
  is taken to be the number of lattice sites `N`.

  The operator acts nontrivially on sites `n1` to `n1 + k - 1` of psi, where
  `0 <= n1 < N`, and is expected to have `2*k` dimensions.
  The first `k` dimensions represent the output and the last `k` dimensions
  represent the input, to be contracted with `psi`.

  Args:
    psi: An `N`-dimensional tensor representing the wavefunction.
    op: Tensor with `2 * k` dimensions. The operator to apply.
    n1: The number of the leftmost site at which to apply the operator.
    pbc: If `True`, use periodic boundary conditions, so that site `N` is
      identified with site `0`. Otherwise, site `N-1` has no neighbors to the
      right.

  Returns:
    psi_final: The result of applying `op` to `psi`.
  """
  n_psi = tensornetwork.Node(psi, backend="tensorflow")
  site_edges = n_psi.get_all_edges()

  site_edges, n_op = _apply_op_network(site_edges, op, n1, pbc)

  n_res = tensornetwork.contract_between(
      n_op, n_psi, output_edge_order=site_edges)

  return n_res.tensor


def _apply_op_network(site_edges, op, n1, pbc=False):
  N = len(site_edges)
  op_sites = len(op.shape) // 2
  n_op = tensornetwork.Node(op, backend="tensorflow")
  for m in range(op_sites):
    target_site = (n1 + m) % N if pbc else n1 + m
    tensornetwork.connect(n_op[op_sites + m], site_edges[target_site])
    site_edges[target_site] = n_op[m]
  return site_edges, n_op


def expval(psi, op, n1, pbc=False):
  """Expectation value of a k-local operator, acting on sites n1 to n1 + k-1.

  In braket notation: <psi|op(n1)|psi>

  The number of dimensions of the tensor representing the wavefunction `psi`
  is taken to be the number of lattice sites `N`.

  Args:
    psi: An `N`-dimensional tensor representing the wavefunction.
    op: Tensor with `2 * k` dimensions. The operator to apply.
    n1: The number of the leftmost site at which to apply the operator.
    pbc: If `True`, use periodic boundary conditions, so that site `N` is
      identified with site `0`. Otherwise, site `N-1` has no neighbors to the
      right.

  Returns:
    expval: The expectation value.
  """
  n_psi = tensornetwork.Node(psi, backend="tensorflow")
  site_edges = n_psi.get_all_edges()

  site_edges, n_op = _apply_op_network(site_edges, op, n1, pbc)

  n_op_psi = n_op @ n_psi

  n_psi_conj = tensornetwork.Node(tf.math.conj(psi), backend="tensorflow")
  for i in range(len(site_edges)):
    tensornetwork.connect(site_edges[i], n_psi_conj[i])

  res = n_psi_conj @ n_op_psi

  return res.tensor


def evolve_trotter(psi,
                   H,
                   step_size,
                   num_steps,
                   euclidean=False,
                   callback=None):
  """Evolve an initial wavefunction psi using a trotter decomposition of H.

  If the evolution is euclidean, the wavefunction will be normalized after
  each step.

  Args:
    psi: An `N`-dimensional tensor representing the initial wavefunction.
    H: A list of `N-1` tensors representing nearest-neighbor operators.
    step_size: The trotter step size.
    num_steps: The number of trotter steps to take.
    euclidean: If `True`, evolve in Euclidean (imaginary) time.
    callback: Optional callback function for monitoring the evolution.

  Returns:
    psi_t: The final wavefunction.
    t: The final time.
  """
  num_sites = len(psi.shape)
  layers = trotter_prepare_gates(H, step_size, num_sites, euclidean)
  return _evolve_trotter_gates(
      psi, layers, step_size, num_steps, euclidean=euclidean, callback=callback)


def _evolve_trotter_gates(psi,
                          layers,
                          step_size,
                          num_steps,
                          euclidean=False,
                          callback=None):
  """Evolve an initial wavefunction psi via gates specified in `layers`.

  If the evolution is euclidean, the wavefunction will be normalized
  after each step.
  """
  t = 0.0
  for i in range(num_steps):
    psi = apply_circuit(psi, layers)
    if euclidean:
      psi = tf.divide(psi, tf.norm(psi))
    t += step_size
    if callback is not None:
      callback(psi, t, i)

  return psi, t


def evolve_trotter_defun(psi,
                         H,
                         step_size,
                         num_steps,
                         euclidean=False,
                         callback=None,
                         batch_size=1):
  """Evolve an initial wavefunction psi using a trotter decomposition of H.

  If the evolution is euclidean, the wavefunction will be normalized after
  each step.

  In this version, `batch_size` steps are "compiled" to a computational graph
  using `defun`, which greatly decreases overhead.

  Args:
    psi: An `N`-dimensional tensor representing the initial wavefunction.
    H: A list of `N-1` tensors representing nearest-neighbor operators.
    step_size: The trotter step size.
    num_steps: The number of trotter steps to take.
    euclidean: If `True`, evolve in Euclidean (imaginary) time.
    callback: Optional callback function for monitoring the evolution.
    batch_size: The number of steps to unroll in the computational graph.

  Returns:
    psi_t: The final wavefunction.
    t: The final time.
  """
  n_batches, rem = divmod(num_steps, batch_size)

  step_size = tf.cast(step_size, psi.dtype)

  num_sites = len(psi.shape)
  layers = trotter_prepare_gates(H, step_size, num_sites, euclidean)

  t = 0.0
  for i in range(n_batches):
    psi, t_b = _evolve_trotter_gates_defun(
        psi, layers, step_size, batch_size, euclidean=euclidean, callback=None)
    t += t_b
    if callback is not None:
      callback(psi, t, (i + 1) * batch_size - 1)

  if rem > 0:
    psi, t_b = _evolve_trotter_gates_defun(
        psi, layers, step_size, rem, euclidean=euclidean, callback=None)
    t += t_b

  return psi, t


#@tf.contrib.eager.defun(autograph=True)
@tf.function(autograph=True)
def _evolve_trotter_gates_defun(psi,
                                layers,
                                step_size,
                                num_steps,
                                euclidean=False,
                                callback=None):
  return _evolve_trotter_gates(
      psi, layers, step_size, num_steps, euclidean=euclidean, callback=callback)


def apply_circuit(psi, layers):
  """Applies a quantum circuit to a wavefunction.

  The circuit consists of a sequence of layers, with each layer consisting
  of non-overlapping gates.

  Args:
    psi: An `N`-dimensional tensor representing the initial wavefunction.
    layers: A sequence of layers. Each layer is a sequence of gates, with
      each index of a layer corresponding to a site in `psi`. The `i`th gate
      of a layer acts on sites `i` to `i + k - 1`, where `k` is the range of
      the gate. Gates may not overlap within a layer.

  Returns:
    psi_t: The final wavefunction.
  """
  num_sites = len(psi.shape)

  n_psi = tensornetwork.Node(psi, backend="tensorflow")
  site_edges = n_psi.get_all_edges()
  nodes = [n_psi]

  for gates in layers:
    skip = 0
    for n in range(num_sites):
      if n < len(gates):
        gate = gates[n]
      else:
        gate = None

      if skip > 0:
        if gate is not None:
          raise ValueError(
              "Overlapping gates in same layer at site {}!".format(n))
        skip -= 1
      elif gate is not None:
        site_edges, n_gate = _apply_op_network(site_edges, gate, n)
        nodes.append(n_gate)

        # keep track of how many sites this gate included
        op_sites = len(gate.shape) // 2
        skip = op_sites - 1

  # NOTE: This may not be the optimal order if transpose costs are considered.
  n_psi = reduce(tensornetwork.contract_between, nodes)
  n_psi.reorder_edges(site_edges)

  return n_psi.tensor
