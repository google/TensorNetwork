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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce
import sys
import tensorflow as tf

import tensornetwork
from trotter import trotter_prepare_gates


def inner(psi1, psi2):
    """Computes the inner product <psi1|psi2>.
    """
    return tf.reduce_sum(tf.conj(psi1) * psi2)


def apply_op(psi, op, n1, maintain_site_ordering=True):
    """Apply a k-local operator, acting on sites n1 to n1 + k-1, to a state.
    """
    net = tensornetwork.TensorNetwork()
    n_psi = net.add_node(psi)
    site_edges = n_psi.get_all_edges()

    net, site_edges, n_op = _apply_op_network(net, site_edges, op, n1)

    n_res = net.contract_between(n_op, n_psi)
    if maintain_site_ordering:
        n_res.reorder_edges(site_edges)

    return n_res.tensor


def _apply_op_network(net, site_edges, op, n1):
    op_sites = len(op.shape) // 2
    n_op = net.add_node(op)
    for m in range(op_sites):
        net.connect(n_op[op_sites + m], site_edges[n1 + m])
        site_edges[n1 + m] = n_op[m]
    return net, site_edges, n_op


def expval(psi, op, n1):
    """Expectation value of a k-local operator, acting on sites n1 to n1 + k-1.
    The operator and state must both be shaped according to the site tensor
    product decomposition.
    """
    op_psi = apply_op(psi, op, n1, maintain_site_ordering=False)
    return inner(psi, op_psi)


def evolve_trotter(psi, H, step_size, num_steps, euclidean=False, callback=None):
    """Evolve an initial state psi using a trotter decomposition of H.
    If the evolution is euclidean, the wavefunction will be normalized after
    each step.
    """
    num_sites = len(psi.shape)
    layers = trotter_prepare_gates(H, step_size, num_sites, euclidean)
    return _evolve_trotter_gates(
        psi, layers, step_size, num_steps,
        euclidean=euclidean,
        callback=callback)


def _evolve_trotter_gates(
    psi, layers, step_size, num_steps,
    euclidean=False, callback=None):
    """Evolve an initial state psi via gates specified in `layers`.
    If the evolution is euclidean, the wavefunction will be normalized after
    each step.
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


def evolve_trotter_defun(
    psi, H, step_size, num_steps,
    euclidean=False,
    callback=None,
    batch_size=1):
    """Evolve an initial state psi using a trotter decomposition of H.
    If the evolution is euclidean, the wavefunction will be normalized after
    each step.

    In this version, `batch_size` steps are "compiled" to a computational graph
    using `defun`, which greatly decreases overhead. 
    """
    n_batches, rem = divmod(num_steps, batch_size)

    step_size = tf.cast(step_size, psi.dtype)

    num_sites = len(psi.shape)
    layers = trotter_prepare_gates(H, step_size, num_sites, euclidean)

    t = 0.0
    for i in range(n_batches):
        psi, t_b = _evolve_trotter_gates_defun(
            psi, layers, step_size, batch_size,
            euclidean=euclidean,
            callback=None)
        t += t_b
        if callback is not None:
            callback(psi, t, (i+1) * batch_size - 1)

    if rem > 0:
        psi, t_b = _evolve_trotter_gates_defun(
            psi, layers, step_size, rem, 
            euclidean=euclidean,
            callback=None)
        t += t_b

    return psi, t


@tf.contrib.eager.defun(autograph=True)
def _evolve_trotter_gates_defun(psi, layers, step_size, num_steps,
    euclidean=False, callback=None):
    return _evolve_trotter_gates(
        psi, layers, step_size, num_steps,
        euclidean=euclidean,
        callback=callback)


def apply_circuit(psi, layers):
    """Applies a quantum circuit to a state.
    The circuit consists of a sequence of layers, with each layer consisting
    of non-overlapping gates.
    """
    num_sites = len(psi.shape)

    net = tensornetwork.TensorNetwork()
    n_psi = net.add_node(psi)
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
                        "Overlapping gates in same layer at site {}!".format(
                            n)
                        )
                skip -= 1
            elif gate is not None:
                net, site_edges, n_gate = _apply_op_network(
                    net, site_edges, gate, n)
                nodes.append(n_gate)

                # keep track of how many sites this gate included
                op_sites = len(gate.shape) // 2
                skip = op_sites - 1

    # NOTE: This may not be the optimal order if transpose costs are considered.
    n_psi = reduce(net.contract_between, nodes)
    n_psi.reorder_edges(site_edges)

    return n_psi.tensor
