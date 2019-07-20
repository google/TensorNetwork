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
"""1-dimensional uniform binary tree tensor network.

Index ordering conventions:
```
iso_012:

  0
  |
(iso)
 / \
1   2
```

iso_021:
```
  0
  |
(iso)
 / \
2   1
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import math
import time
import tensornetwork





def build(x):
  """Dummy function for building a python function into a graph."""
  return x


def _ascend_partial(op, iso):
  """Contract an operator with the rightmost index of an isometry.

  For 012 (021) index ordering, this is equivalent to contracting with the
  physical right (left). This is "half" of the operation needed to ascend
  an operator via the isometry.
  To complete, use `_complete_partial_ascend()`.

  Cost: D^4.

  Args:
    op: The operator to ascend (a matrix). Dimensions must match the
      dimensions of the lower indices of the isometry.
    iso: The isometry (a rank-3 tensor).

  Returns:
    The result of contracting `op` with `iso`.
  """
  return tensornetwork.ncon([iso, op], [(-1, -2, 1), (-3, 1)])


def _complete_partial_ascend(iso_op, iso):
  """Complete a partial operator ascension performed by `_ascend_partial()`.

  This contracts with the conjugated isometry.

  Cost: D^4.

  Args:
    iso_op: Operator contracted with the isometry (result of 
      `_ascend_partial()`).
    iso: The isometry (a rank-3 tensor).

  Returns:
    The ascended operator.
  """
  return tensornetwork.ncon([conj(iso), iso_op], [(-1, 1, 2), (-2, 1, 2)])


def _ascend_op_2site_to_1site_partial(mpo_2site, iso_021):
  """Contract a 2-site MPO with a single isometry.

  Produces an ascended (1-site) operator after completion via
  `_complete_partial_ascend()`.

  Cost: D^4.

  Args:
    mpo_2site: The 2-site MPO consisting of two lists of the same length (the
      MPO bond dimension), one for each site, of 1-site operators.
    iso_021: The isometry (a rank-3 tensor) with "021" ordering.

  Returns:
    The result of contracting the operator with the isometry.
  """
  op2L, op2R = mpo_2site

  M = len(op2L)  # MPO bond dimension

  terms = []
  for m in range(M):
    # permute result to 012 order: M mild transposes
    iso_op_mpo_L_012 = tensornetwork.ncon([iso_021, op2L[m]], [(-1, -3, 1),
                                                               (-2, 1)])

    terms.append(_ascend_partial(op2R[m], iso_op_mpo_L_012))
  iso_op_2site_012 = sum(terms)

  return iso_op_2site_012


def _ascend_uniform_op_to_1site_partial(op_1site, mpo_2site, iso_012, iso_021):
  """Contract a uniform 2-site operator with a single isometry.

  A "uniform 2-site operator" means an operator that is a sum of of two equal
  1-site terms and a single 2-site MPO term:
    "op = op_1site(0) + op_1site(1) + mpo_2site"

  Produces an ascended (1-site) operator after completion via
  `_complete_partial_ascend()`.

  Cost: D^4.

  Args:
    op_1site: The 1-site term.
    mpo_2site: The 2-site MPO term.
    iso_021: The isometry (a rank-3 tensor) with "021" ordering.

  Returns:
    res_012: The result of contracting the operator with the isometry,
      012 ordering.
    res_021: The result of contracting the operator with the isometry,
      021 ordering.
  """
  iso_op_2site_012 = _ascend_op_2site_to_1site_partial(mpo_2site, iso_021)
  iso_op_1site_R_012 = _ascend_partial(op_1site, iso_012)
  iso_op_1site_L_021 = _ascend_partial(op_1site, iso_021)
  return iso_op_2site_012 + iso_op_1site_R_012, iso_op_1site_L_021


def ascend_op_1site_to_1site_R(op_1site, iso_012):
  """Ascends a 1-site operator from the right of an isometry.

  Note: If called with an isometry using "021" ordering, this ascends from
    the left instead.

  Args:
    op_1site: The 1-site operator (a matrix).
    iso_012: The isometry (a rank-3 tensor).

  Returns:
    The ascended operator.
  """
  return _complete_partial_ascend(_ascend_partial(op_1site, iso_012), iso_012)


def ascend_op_1site_to_1site_L(op_1site, iso_012):
  """Ascends a 1-site operator from the left of an isometry.

  Args:
    op_1site: The 1-site operator (a matrix).
    iso_012: The isometry (a rank-3 tensor).

  Returns:
    The ascended operator.
  """
  return ascend_op_1site_to_1site_R(op_1site, transpose(iso_012, (0,2,1)))


def ascend_uniform_op_to_1site(op_1site, mpo_2site, iso_012, iso_021):
  """Ascends a uniform 2-site operator through an isometry.

  A "uniform 2-site operator" means an operator that is a sum of of two equal
  1-site terms and a single 2-site MPO term:
    "op = op_1site(0) + op_1site(1) + mpo_2site"

  Args:
    op_1site: The 1-site term.
    mpo_2site: The 2-site MPO term.
    iso_012: The isometry (a rank-3 tensor) with "012" ordering.
    iso_021: The isometry (a rank-3 tensor) with "021" ordering.

  Returns:
    The ascended operator.
  """
  terms_012, iso_op_1site_L_021 = _ascend_uniform_op_to_1site_partial(
      op_1site, mpo_2site, iso_012, iso_021)

  res = _complete_partial_ascend(iso_op_1site_L_021, iso_021)
  res += _complete_partial_ascend(terms_012, iso_012)

  return res


def ascend_op_2site_to_1site(mpo_2site, iso_012, iso_021):
  """Ascends a 2-site MPO through a single isometry.

  Args:
    mpo_2site: The 2-site MPO.
    iso_012: The isometry (a rank-3 tensor) with "012" ordering.
    iso_021: The isometry (a rank-3 tensor) with "021" ordering.

  Returns:
    The ascended operator, now a 1-site operator.
  """
  iso_op_2site_012 = _ascend_op_2site_to_1site_partial(mpo_2site, iso_021)
  return _complete_partial_ascend(iso_op_2site_012, iso_012)


def ascend_op_2site_to_2site(mpo_2site, iso_012, iso_021):
  """Ascends a 2-site MPO through a pair of isometries.

  Given a pair of neighboring isometries, each with two lower indices,
  ascends a 2-site MPO through the middle two indices to form a new 2-site
  MPO.

  Args:
    mpo_2site: The 2-site MPO.
    iso_012: The isometry (a rank-3 tensor) with "012" ordering.
    iso_021: The isometry (a rank-3 tensor) with "021" ordering.

  Returns:
    The ascended operator, a 2-site MPO.
  """
  def _ascend(op, iso, iso_conj):
    return tensornetwork.ncon([iso_conj, op, iso], [(-1, 3, 1), (1, 2),
                                                    (-2, 3, 2)])

  op2L, op2R = mpo_2site

  M = len(op2L)

  iso_021_conj = conj(iso_021)
  op_asc_R = []
  for m in range(M):
    op_asc_R.append(_ascend(op2R[m], iso_021, iso_021_conj))

  iso_012_conj = conj(iso_012)
  op_asc_L = []
  for m in range(M):
    op_asc_L.append(_ascend(op2L[m], iso_012, iso_012_conj))

  return op_asc_L, op_asc_R


def ascend_uniform_op_local(op_1site, mpo_2site, iso_012, iso_021):
  """Ascends a globally uniform operator through a periodic layer of isometries.

  The operator is assumed to consist of a sum of equal 1-site terms, the same
  on every site, plus a sum of 2-site MPOs, also the same for each pair of
  neighboring sites.

  This is ascended though a uniform layer of isometries to produce an
  ascended of the same form.

  It is assumed that the layer of isometries consists of more than isometry.
  If this is not the case, use `ascend_uniform_op_local_top()`.

  Args:
    op_1site: The 1-site term.
    mpo_2site: The 2-site MPO term.
    iso_012: The isometry (a rank-3 tensor) with "012" ordering.
    iso_021: The isometry (a rank-3 tensor) with "021" ordering.

  Returns:
    op_1site: The 1-site component of the ascended operator.
    op_2site: The 2-site MPO component of the ascended operator.
  """
  op_1site = ascend_uniform_op_to_1site(op_1site, mpo_2site, iso_012, iso_021)
  mpo_2site = ascend_op_2site_to_2site(mpo_2site, iso_012, iso_021)
  return op_1site, mpo_2site


ascend_uniform_op_local_graph = build(ascend_uniform_op_local)


def ascend_uniform_op_local_top(op_1site, mpo_2site, iso_012, iso_021):
  """Ascends a globally uniform operator through the top tensor of a tree.

  See `ascend_uniform_op_local()`. This ascends a globally uniform operator
  through a periodic layer of isometries consisting of only a single isometry
  as occurs at the top of a tree tensor network.

  The result is a 1-site operator.

  Args:
    op_1site: The 1-site term.
    mpo_2site: The 2-site MPO term.
    iso_012: The isometry (a rank-3 tensor) with "012" ordering.
    iso_021: The isometry (a rank-3 tensor) with "021" ordering.

  Returns:
    The ascended operator, a 1-site operator.
  """
  mpo_2site = add_mpos_2site(mpo_2site, reflect_mpo_2site(mpo_2site))
  op_1site = ascend_uniform_op_to_1site(op_1site, mpo_2site, iso_012, iso_021)
  return op_1site


ascend_uniform_op_local_top_graph = build(ascend_uniform_op_local_top)


def ascend_uniform_op_local_many(op_1site, mpo_2site, isos):
  """Ascends a globally uniform operator through many layers.

  Returns intermediate results.
  See `ascend_uniform_op_local()`.

  Args:
    op_1site: The 1-site term.
    mpo_2site: The 2-site MPO term.
    isos: List of isometries, each representing a uniform layer through which
      the operator is to be ascended.

  Returns:
    A list of pairs of 1-site and MPO terms. Each entry `i` is the result of
    ascending through the layers defined by `isos[:i+1]`.
  """
  ops = []
  for l in range(len(isos)):
    op_1site, mpo_2site = ascend_uniform_op_local(op_1site, mpo_2site, *isos[l])
    ops.append(op_1site, mpo_2site)
  return ops


def ascend_uniform_MPO_to_top(mpo_tensor_dense, isos_012):
  """Ascends a globally uniform MPO to the top of a tree.

  Unlike the 2-site MPOs used elsewhere, this takes a dense MPO tensor of
  rank 4 with the following ordering convention:

        3
        |
      0--m--1
        |
        2

  The bottom and top indices are the "left" and "right" indices of 1-site
  operators. The left and right indices are the MPO bond indices.

  Args:
    mpo_tensor_dense: The MPO tensor.
    isos_012: List of isometries with 012 ordering, defining the tree.

  Returns:
    A 1-site operator acting on the top space of the tree.
  """
  L = len(isos_012)
  for l in range(L):
    # NOTE: There is no attempt to be economical with transpose here!
    mpo_tensor_dense = tensornetwork.ncon(
        [isos_012[l],
         conj(isos_012[l]), mpo_tensor_dense, mpo_tensor_dense],
        [(-4, 3, 1), (-3, 4, 5), (2, -2, 5, 1), (-1, 2, 4, 3)])
  op = tensornetwork.ncon([mpo_tensor_dense], [(1, 1, -1, -2)])
  return op


def descend_state_1site_R(state_1site, iso_012):
  """Descends a 1-site density matrix though a single isometry to the right.

  Produces a 1-site density matrix on the right site by tracing out the left
  site.

  Cost: D^4.

  Args:
    state_1site: The 1-site density matrix.
    iso_012: Isometry (rank-3 tensor) with 012 ordering.

  Returns:
    Descended 1-site density matrix.
  """
  return tensornetwork.ncon(
    [iso_012, state_1site, conj(iso_012)],
    [(2, 3, -1), (2, 1), (1, 3, -2)])


def descend_state_1site_L(state_1site, iso_021):
  """Descends a 1-site density matrix though a single isometry to the left.

  Produces a 1-site density matrix on the left site by tracing out the right
  site.

  Cost: D^4.

  Args:
    state_1site: The 1-site density matrix.
    iso_021: Isometry (rank-3 tensor) with 021 ordering.

  Returns:
    Descended 1-site density matrix.
"""
  return descend_state_1site_R(state_1site, iso_021)


def descend_state_1site(state_1site, iso_012, iso_021):
  """Average descended 1-site.

  The average of `descend_state_1site_R()` and `descend_state_1site_L()`.

  Cost: D^4.

  Args:
    state_1site: The 1-site density matrix.
    iso_012: Isometry (rank-3 tensor) with 012 ordering.
    iso_021: The same isometry, but with 021 ordering.

  Returns:
    Descended 1-site density matrix.
  """
  state_1L = descend_state_1site_L(state_1site, iso_021)
  state_1R = descend_state_1site_R(state_1site, iso_012)
  return 0.5 * (state_1L + state_1R)


def correlations_2pt_1s(isos_012, op):
  """Computes a two-point correlation function for a 1-site operator `op`.

  Args:
    isos_012: List of isometries defining the uniform tree.
    op: The 1-site operator (matrix).

  Returns:
    cf_transl_avg: Translation-averaged correlation function (as a vector).
    cf_1: Partially translation-averaged correlation function (as a vector).
  """
  if len(op.shape) != 2:
    raise ValueError("Operator must be a matrix.")
  nsites = 2**len(isos_012)
  states = all_states_1site_graph(isos_012)
  expval_sq = trace(states[0] @ op)**2
  twopoints = {}
  asc_ops = {0: op}
  for l in range(len(isos_012)):
    iso_012 = isos_012[l]
    iso_021 = transpose(iso_012, (0,2,1))
    # Compute all two-point functions available at this level
    for (site1, asc_op1) in asc_ops.items():
      for (site2, asc_op2) in asc_ops.items():
        asc_op12 = ascend_op_2site_to_1site(
          ([asc_op1], [asc_op2]),
          iso_012,
          iso_021)
        site2 += 2**l
        twopoints[(site1, site2)] = (
          trace(asc_op12 @ states[l+1]) - expval_sq)
    if l < len(isos_012) - 1:
      asc_ops_new = {}
      for (site, asc_op) in asc_ops.items():
        asc_opL = ascend_op_1site_to_1site_R(asc_op, iso_021)  # R is correct.
        asc_opR = ascend_op_1site_to_1site_R(asc_op, iso_012)
        asc_ops_new[site] = asc_opL
        asc_ops_new[site + 2**l] = asc_opR
      asc_ops = asc_ops_new
  corr_func = {}
  for ((site1, site2), val) in twopoints.items():
    dist = abs(site1 - site2)
    try:
      corr_func[dist].append(val)
    except KeyError:
      corr_func[dist] = [val]
  # Final translation averaging
  for (dist, vals) in corr_func.items():
    corr_func[dist] = sum(vals) / len(vals)
  dists = sorted(corr_func.keys())
  cf_transl_avg = convert_to_tensor([corr_func[d] for d in dists])
  cf_1 = convert_to_tensor([twopoints[(0,i)] for i in range(1,nsites)])
  return cf_transl_avg, cf_1


def reflect_mpo_2site(mpo_2site):
  """Spatial reflection of a 2-site MPO.
  """
  return tuple(reversed(mpo_2site))


def add_mpos_2site(mpo1, mpo2):
  """Sum of two 2-site MPOs acting on the same pair of sites.
  """
  return (mpo1[0] + mpo2[0], mpo1[1] + mpo2[1])


def opt_energy_env_2site(isos_012, h_mpo_2site, states_1site_above):
  """Computes 2-site Hamiltonian contributions to the isometry environment.

  This always computes the environment contribution for the isometry in the
  first entry of `isos_012`. To compute environments for higher levels in a
  three, supply data for the truncated tree: For level `l` call with
  `isos_012[l:]` and the corresponding hamiltonian and states.

  Args:
    isos_012: The isometries defining the tree tensor network.
    h_mpo_2site: The 2-site term of the uniform Hamiltonian for the bottom
      of the network defined in `isos_012`.
    states_1site_above: 1-site translation-averaged density matrices for each
      level above the bottom of the network defined in `isos_012`.

  Returns:
    Environment tensor (rank 3).
  """
  def _ascend_op_2site_to_2site_many(mpo_2site, isos):
    ops = []
    for l in range(len(isos)):
      mpo_2site = ascend_op_2site_to_2site(mpo_2site, *isos[l])
      ops.append(mpo_2site)
    return ops

  def _mpo_with_state(iso_012, iso_021, h_mpo_2site, state_1site):
    """Contract a 2-site MPO with a 1-site descended state. O(D^4)"""
    h2L, h2R = h_mpo_2site

    envL = [
        tensornetwork.ncon(
            [state_1site, iso_021, h, conj(iso_012)],
            [(1, 3), (1, -1, 2), (4, 2), (3, 4, -2)])  # one transpose required
        for h in h2L
    ]

    envR = [
        tensornetwork.ncon(
            [state_1site, iso_012, h, conj(iso_021)],
            [(1, 3), (1, -1, 2), (4, 2), (3, 4, -2)])  # one transpose required
        for h in h2R
    ]

    return envL, envR

  def _descend_energy_env_L(env, iso_021):
    return [descend_state_1site_L(e, iso_021) for e in env]

  def _descend_energy_env_R(env, iso_012):
    return [descend_state_1site_R(e, iso_012) for e in env]

  isos_wt = isos_with_transposes(isos_012)
  iso_012, iso_021 = isos_wt[0]
  isos_wt_above = isos_wt[1:]
  levels_above = len(isos_wt_above)

  # Ascend two-site Hamiltonian terms to the bottom of the final isometry
  h2s_above = _ascend_op_2site_to_2site_many(h_mpo_2site, isos_wt)

  # hamiltonian with isometry opposite the gap
  h2L, h2R = h_mpo_2site
  iso_h2R_012 = [
      tensornetwork.ncon([iso_021, h], [(-1, -3, 1), (-2, 1)]) for h in h2R
  ]  # transpose to 012
  iso_h2L_012 = [
      tensornetwork.ncon([iso_012, h], [(-1, -2, 1), (-3, 1)]) for h in h2L
  ]

  def _compute_env(lvl, reflect=False):
    # TODO: Could shorten this a bit by doing only left or right at one time
    h2 = h2s_above[lvl]
    if reflect:
      h2 = reflect_mpo_2site(h2)

    envL, envR = _mpo_with_state(*isos_wt_above[lvl], h2,
                                 states_1site_above[lvl])

    # descend envs back down to the level of the gap
    for lvl2 in reversed(range(lvl)):
      iso_012_l2, iso_021_l2 = isos_wt_above[lvl2]
      if reflect:
        envR = _descend_energy_env_L(envR, iso_021_l2)
        envL = _descend_energy_env_R(envL, iso_012_l2)
      else:
        envL = _descend_energy_env_L(envL, iso_021_l2)
        envR = _descend_energy_env_R(envR, iso_012_l2)

    if reflect:
      iso_h2_L, iso_h2_R = iso_h2R_012, iso_h2L_012
    else:
      iso_h2_L, iso_h2_R = iso_h2L_012, iso_h2R_012

    # contract with the hamiltonian + isometry opposite the gap
    envL = sum(
        tensornetwork.ncon([eL, ihR], [(1, -1), (1, -2, -3)])
        for eL, ihR in zip(envL, iso_h2_R))

    envR = sum(
        tensornetwork.ncon([eR, ihL], [(1, -1), (1, -2, -3)])
        for eR, ihL in zip(envR, iso_h2_L))

    # weight each term according to the number of occurrences
    # in the translation-invariant tree
    weight = 1 / 2.0**(lvl + 1)
    return (envL + envR) * weight, weight

  weightsum = 0.0
  env_total = []
  for lvl in range(levels_above):
    env, weight = _compute_env(lvl)
    weightsum += weight
    env_total.append(env)

  # Now compute the boundary term
  env, weight = _compute_env(levels_above - 1, reflect=True)
  weightsum += weight
  env_total.append(env)

  env_total = sum(env_total)

  assert weightsum == 1.0

  return env_total


def opt_energy_env_1site(iso_012, h_op_1site, h_mpo_2site, state_1site):
  """Computes 1-site Hamiltonian contributions to the isometry environment.

  Args:
    iso_012: The isometry whose environment is desired.
    h_op_1site: The 1-site term of the uniform Hamiltonian for the bottom
      of the layer defined by the isometry.
    h_mpo_2site: The 2-site term of the uniform Hamiltonian for the bottom
      of the layer defined by the isometry.
    state_1site: 1-site translation-averaged density matrix for the top of
      the layer defined by the isometry.

  Returns:
    Environment tensor (rank 3).
  """
  iso_021 = transpose(iso_012, (0, 2, 1))
  terms_012, terms_021 = _ascend_uniform_op_to_1site_partial(
    h_op_1site, h_mpo_2site, iso_012, iso_021)
  terms = terms_012 + transpose(terms_021, (0, 2, 1))
  env = tensornetwork.ncon([state_1site, terms], [(1, -1), (1, -2, -3)])
  return env


def opt_energy_env(isos_012,
                   h_op_1site,
                   h_mpo_2site,
                   states_1site_above):
  """Computes the isometry environment for the energy expectation value network.

  This always computes the environment contribution for the isometry in the
  first entry of `isos_012`. To compute environments for higher levels in a
  three, supply data for the truncated tree: For level `l` call with
  `isos_012[l:]` and the corresponding hamiltonian terms and states.

  Args:
    isos_012: The isometries defining the tree tensor network.
    h_op_1site: The 1-site term of the uniform Hamiltonian for the bottom
      of the network defined in `isos_012`.
    h_mpo_2site: The 2-site term of the uniform Hamiltonian for the bottom
      of the network defined in `isos_012`.
    states_1site_above: 1-site translation-averaged density matrices for each
      level above the bottom of the network defined in `isos_012`.
    envsq_dtype: Used to specify a different dtype for the computation of the
      squared environment, if used.

  Returns:
    Environment tensor (rank 3).
  """
  if len(isos_012) == 1:  # top of tree
    h_mpo_2site = add_mpos_2site(h_mpo_2site, reflect_mpo_2site(h_mpo_2site))
    env = opt_energy_env_1site(isos_012[0], h_op_1site, h_mpo_2site,
                               states_1site_above[0])
  else:
    env1 = opt_energy_env_1site(isos_012[0], h_op_1site, h_mpo_2site,
                                states_1site_above[0])
    env2 = opt_energy_env_2site(isos_012, h_mpo_2site, states_1site_above[1:])
    env = env1 + env2

  return env


opt_energy_env_graph = build(opt_energy_env)


def _uinv_decomp(X_sq, cutoff=0.0, decomp_mode="eigh"):
  """Computes an "inverse" from the square of a rectangular matrix.
  The matrix returned is the inverse up to a unitary transformation. So not
  really an inverse at all.

  Args:
    X_sq: A positive Hermitian matrix (the square of rectangular matrix).
    cutoff: Threshold for pseudo-inversion.
    decomp_mode: Can be "eigh" of "svd". The former should be slightly faster.

  Returns:
    X_uinv: An "inverse" of the original rectangular matrix.
    s: The singular values of the square root of X_sq.
  """
  if decomp_mode == "svd":
    # hermitian, positive matrix, so eigvals = singular values
    e, v, _ = svd(X_sq)
  elif decomp_mode == "eigh":
    e, v = eigh(X_sq)
    e = to_real(e)  # The values here should be real anyway
  else:
    raise ValueError("Invalid decomp_mode: {}".format(decomp_mode))

  s = sqrt(e)  # singular values of the square root of X_sq
  # NOTE: Negative values are always due to precision problems.
  # NOTE: Inaccuracies here mean the final tensor is not exactly isometric!
  e_pinvsqrt = where(e <= cutoff, zeros_like(e), 1 / s)

  e_pinvsqrt_mat = diag(cast(e_pinvsqrt, v.dtype))
  X_uinv = matmul(v @ e_pinvsqrt_mat, v, adjoint_b=True)
  return X_uinv, s


def _iso_from_envsq_decomp(env,
                  cutoff=0.0,
                  decomp_mode="eigh",
                  decomp_device=None,
                  envsq_dtype=None):
  """Computes a new optimal isometry from the square of the environment tensor.

  The precision of the result is the square root of the working precision,
  so the working precision for this operation can be specified separately via
  the `envsq_dtype` argument. A different device may also be specified, in case
  the current device does not support the required precision or operations.
  """
  if envsq_dtype is not None:
    env = cast(env, envsq_dtype)
  with device(decomp_device):
    env_sq = tensornetwork.ncon([env, conj(env)], [(-1, 1, 2), (-2, 1, 2)])
    env_uinv, s = _uinv_decomp(env_sq, cutoff, decomp_mode, decomp_device)
    iso_012_new = tensornetwork.ncon([env_uinv, env], [(-1, 1), (1, -2, -3)])
  if envsq_dtype is not None:
    iso_012_new = cast(iso_012_new, dtype)
  return iso_012_new, s


_iso_from_envsq_decomp_graph = build(_iso_from_envsq_decomp)


def _energy_expval_env(isos_012, h_op_1site, h_mpo_2site, states_1site_above):
  """Computes the energy using the environments. For testing.
  """
  if len(isos_012) == 1:  # top of tree
    h_mpo_2site = add_mpos_2site(h_mpo_2site, reflect_mpo_2site(h_mpo_2site))
    env = opt_energy_env_1site(isos_012[0], h_op_1site, h_mpo_2site,
                               states_1site_above[0])
  else:
    env1 = opt_energy_env_1site(isos_012[0], h_op_1site, h_mpo_2site,
                                states_1site_above[0])
    env2 = opt_energy_env_2site(isos_012, h_mpo_2site, states_1site_above[1:])
    env = env1 + env2 / 2
    # NOTE: There are *two* environments for each Ham. term spanning two
    #       isometries. To get the correct energy we must divide env2 by 2.
  nsites = 2**(len(isos_012) - 1)
  return tensornetwork.ncon([conj(isos_012[0]), env], [(1, 2, 3),
                                                          (1, 2, 3)]) * nsites


def _iso_from_svd(u, vh):
  return tensornetwork.ncon([u, vh], [(-1, 1), (1, -2, -3)])


_iso_from_svd_graph = build(_iso_from_svd)


def _iso_from_svd_decomp(env, decomp_device=None):
  """Isometry update using SVD of environment.
  """
  with device(decomp_device):
    env_r = reshape(env, (env.shape[0], -1))
    s, u, v = svd(env_r)
    vh = adjoint(v)
    vh = reshape(vh, (vh.shape[0], env.shape[1], env.shape[2]))
    iso_new = _iso_from_svd(u, vh)
    return iso_new, s


_iso_from_svd_decomp_graph = build(_iso_from_svd_decomp)


def _iso_from_svd_decomp_scipy(env):
  """Isometry update using SVD of environment using scipy's SVD.
  When scipy is built with the MKL, this is the MKL SVD, which currently 
  parallelizes better than TensorFlow's SVD on CPU.
  """
  env = to_numpy(env)
  env_r = env.reshape((env.shape[0], -1))
  u, s, vh = svd_np(env_r, full_matrices=False)
  u = convert_to_tensor(u)
  s = convert_to_tensor(s)
  vh = vh.reshape((vh.shape[0], env.shape[1], env.shape[2]))
  vh = convert_to_tensor(vh)
  return u, s, vh


def opt_energy_layer_once(isos_012,
                          h_op_1site,
                          h_mpo_2site,
                          states_1site_above,
                          graphed=False,
                          decomp_mode="svd_full_iso",
                          decomp_device=None,
                          envsq_dtype=None,
                          timing=False):
  """Updates a layer of the tree via a linearized energy optimization.

  Args:
    isos_012: The isometries for the tree, beginning at the layer to be
      updated.
    h_op_1site: The 1-site term of the uniform Hamiltonian for the bottom
      of the network defined in `isos_012`.
    h_mpo_2site: The 2-site term of the uniform Hamiltonian for the bottom
      of the network defined in `isos_012`.
    states_1site_above: 1-site translation-averaged density matrices for each
      level above the bottom of the network defined in `isos_012`.
    graphed: Whether to build computational graphs of certain groups of
      operations. This can speed up computation, but may increase memory usage.
    decomp_mode: The decomposition used to update the isometries.
    decomp_device: Device on which to perform the decomposition.
    envsq_dtype: Used to specify a different dtype for the computation of the
      squared environment and its decomposition, if used.
    timing: Whether to gather timing information (decomps vs. environments).

  Returns:
    iso_012_new: Updated isometry for the current layer.
    s: Singular values of the environment.
    t_env: Time spent computing the environment (only returned if timing is
      True).
    t_decomp: Time spent computing the decomposition (only returned if timing
      is True).
  """
  t0 = time.time()
  if graphed:
    env = opt_energy_env_graph(
        isos_012,
        h_op_1site,
        h_mpo_2site,
        states_1site_above)
  else:
    env = opt_energy_env(
        isos_012,
        h_op_1site,
        h_mpo_2site,
        states_1site_above)

  if timing and executing_eagerly():
    # Hack to ensure values on GPU are ready. Only works for TensorFlow.
    to_numpy(env_sq[0,0])

  t_env = time.time() - t0

  t0 = time.time()
  if decomp_mode == "svd_full_iso":
    if graphed:
      iso_012_new, s = _iso_from_svd_decomp_graph(
        env, decomp_device=decomp_device)
    else:
      iso_012_new, s = _iso_from_svd_decomp(env, decomp_device=decomp_device)
  elif decomp_mode == "svd_full_iso_scipy":
    u, s, vh = _iso_from_svd_decomp_scipy(env)
    if graphed:
      iso_012_new = _iso_from_svd_graph(u, vh)
    else:
      iso_012_new = _iso_from_svd(u, vh)
  elif decomp_mode == "eigh":
    if graphed:
      iso_012_new, s = _iso_from_envsq_decomp_graph(
          env,
          decomp_mode=decomp_mode,
          decomp_device=decomp_device,
          envsq_dtype=envsq_dtype)
    else:
      iso_012_new, s = _iso_from_envsq_decomp(
          env,
          decomp_mode=decomp_mode,
          decomp_device=decomp_device,
          envsq_dtype=envsq_dtype)
  else:
    raise ValueError("Invalid decomp mode: {}".format(decomp_mode))

  if timing and executing_eagerly():
    # Hack to ensure values on GPU are ready. Only works for TensorFlow.
    to_numpy(iso_012_new[0,0,0])

  t_decomp = time.time() - t0

  if timing:
    return iso_012_new, s, t_env, t_decomp
  return iso_012_new, s


opt_energy_layer_once_graph = build(opt_energy_layer_once)


def opt_energy_layer(isos_012,
                     h_op_1site,
                     h_mpo_2site,
                     states_1site_above,
                     itr,
                     graphed=False,
                     graph_level=None,
                     decomp_mode="eigh",
                     decomp_device=None,
                     envsq_dtype=None,
                     timing=False):
  """Updates a layer of tree by doing several linearized energy optimizations.

  Args:
    isos_012: The isometries for the tree, beginning at the layer to be updated.
    h_op_1site: The 1-site term of the uniform Hamiltonian for the bottom
      of the network defined in `isos_012`.
    h_mpo_2site: The 2-site term of the uniform Hamiltonian for the bottom
      of the network defined in `isos_012`.
    states_1site_above: 1-site translation-averaged density matrices for each
      level above the bottom of the network defined in `isos_012`.
    itr: How many linearized updates to do.
    graphed: Whether to build computational graphs of certain groups of
      operations. This can speed up computation, but may increase memory usage.
    graph_level: If "sweep", use a single graph for the entire linearized
      update. Otherwise use separate graphs for decomp. and environment.
    decomp_mode: The decomposition used to update the isometries.
    decomp_device: Device on which to perform the decomposition.
    envsq_dtype: Used to specify a different dtype for the computation of the
      squared environment and its decomposition, if used.
    timing: Whether to gather timing information (decomps vs. environments).

  Returns:
    iso_012: Updated isometry for the current layer.
    s: Singular values of the environment (from the final iteration).
    t_env: Average time spent computing the environment (only returned if
      timing is True).
    t_decomp: Average time spent computing the decomposition (only returned
      if timing is True).
  """
  shp = isos_012[0].shape
  if shp[0] == shp[1] * shp[2]:  # unitary, nothing to optimise
    return isos_012[0]

  iso_012 = isos_012[0]
  s = None
  tes, tds = 0.0, 0.0
  for _ in range(itr):
    if graph_level == "sweep":
      if timing:
        raise ValueError("Timing data not available with graph_level 'sweep'")
      iso_012, s = opt_energy_layer_once_graph(
          isos_012,
          h_op_1site,
          h_mpo_2site,
          states_1site_above,
          graphed=False,
          decomp_mode=decomp_mode,
          decomp_device=decomp_device,
          envsq_dtype=envsq_dtype,
          timing=False)
    else:
      res = opt_energy_layer_once(
          isos_012,
          h_op_1site,
          h_mpo_2site,
          states_1site_above,
          graphed=graphed,
          decomp_mode=decomp_mode,
          decomp_device=decomp_device,
          envsq_dtype=envsq_dtype,
          timing=timing)
      iso_012, s = res[:2]
      if timing:
        te, td = res[2:]
        tes += te
        tds += td

  if timing:
    return iso_012, s, tes / itr, tds / itr
  return iso_012, s


def all_states_1site(isos_012):
  """Compute 1-site reduced states for all levels of a tree tensor network.

  Args:
    isos_012: The isometries definiting the tree tensor network (bottom to top).

  Returns:
    states: L+1 1-site reduced states, where L is the number of layers in the
      tree. Bottom to top ordering.
  """
  states = [eye(isos_012[-1].shape[0], dtype=isos_012[0][0].dtype)]
  for l in reversed(range(len(isos_012))):
    iso_021 = transpose(isos_012[l], (0, 2, 1))
    states.append(descend_state_1site(states[-1], isos_012[l], iso_021))
  return states[::-1]


all_states_1site_graph = build(all_states_1site)


def entanglement_specs_1site(isos_012):
  """1-site entanglement spectra for all levels of a tree tensor network.

  Here, "entanglement spectrum" means the spectrum of the reduced density
  matrix (rather than the log of that spectrum).

  Args:
    isos_012: The isometries definiting the tree tensor network (bottom to top).

  Returns:
    specs: L 1-site entanglement spectra, where L is the number of layers in
      the tree. Bottom to top ordering.
  """
  specs = []
  state = eye(isos_012[-1].shape[0], dtype=isos_012[0][0].dtype)
  for l in reversed(range(len(isos_012))):
    iso_021 = transpose(isos_012[l], (0, 2, 1))
    state = descend_state_1site(state, isos_012[l], iso_021)
    e = eigvalsh(state)
    e = to_real(e)
    specs.append(e)
  return specs[::-1]


def entropies_from_specs(specs):
  """Compute entanglement entropies from a list of entanglement spectra.

  Here, "entanglement spectrum" means the spectrum of the reduced density
  matrix (rather than the log of that spectrum) and "entanglement entropy"
  means the von Neumann entropy using base 2 for the logarithm.

  Negative entries int he entanglement spectrum are treated as zeros.

  Args:
    specs: List of entanglement spectra.

  Returns:
    entropies: List of entanglement entropies.
  """
  entropies = []
  for spec in specs:
    spec = to_numpy(spec)
    x = spec * np.log2(spec)
    x[np.isnan(x)] = 0.0  # ignore zero or negative eigenvalues
    S = -np.sum(x)
    entropies.append(S)
  return entropies


def random_isometry_cheap(D1, D2, dtype, decomp_mode="eigh"):
  """Generate a random isometric matrix of dimension D1 x D2 more cheaply.

  This uses a decomposition of the square of a random matrix to generate an
  isometry. Since the initial matrix is random, the singular values of its
  square should not be small, and the halving of the precision due to the
  squaring should not be cause significant violations of the isometry property.

  We require D1 <= D2.

  Args:
    D1: Left dimension.
    D2: Right dimension.
    dtype: Element type.
    decomp_mode: "eigh" or "svd".

  Returns:
    V: An isometry.
  """
  if not D1 <= D2:
    raise ValueError("The left dimension must be <= the right dimension.")
  A = random_normal_mat(D1, D2, dtype)
  A_inv, _ = _uinv_decomp(matmul(A, A, adjoint_b=True), decomp_mode=decomp_mode)
  return A_inv @ A


def random_isometry(D1, D2, dtype):
  """Generate a random isometric matrix of dimension D1 x D2.

  We require D1 <= D2.

  Args:
    D1: Left dimension.
    D2: Right dimension.
    dtype: Element type.

  Returns:
    V: An isometry.
  """
  if not D1 <= D2:
    raise ValueError("The left dimension must be <= the right dimension.")
  A = random_normal_mat(D2, D1, dtype)
  Q, R = qr(A)
  r = diag_part(R)
  L = diag(r / cast(abvals(r), dtype))
  return transpose(Q @ L)


def random_tree_tn_uniform(Ds, dtype, top_rank=1):
  """Generate a random tree tensor network.

  Args:
    Ds: List of bond dimensions, one for each layer in the tree. The first
      entry is the "physical dimension".
    dtype: Data dtype for the tensor elements.
    top_rank: The top dimension of the tree. A value of 1 produces a pure
      state. A value > 1 produces an equal mixture of normalized pure states.

  Returns:
    isos: List of random isometries defining the tree tensor network.
  """
  num_layers = len(Ds)
  Ds = Ds + [top_rank]
  isos = []
  for j in range(num_layers):
    if Ds[j + 1] == Ds[j]**2:
      iso = eye(Ds[j + 1], dtype=dtype)
    else:
      iso = random_isometry(Ds[j + 1], Ds[j]**2, dtype)
    iso = reshape(iso, (Ds[j + 1], Ds[j], Ds[j]))
    isos.append(iso)
  return isos


def expand_bonds(isos, new_Ds, new_top_rank=None):
  """Expand the bond dimension of a tree tensor network.

  Inserts random isometry pairs on the bonds of the tree as necessary to
  increase the bond dimension as requested. The state represented is not
  changed by this operation.

  Args:
    isos: List of isometries defining the tree.
    Ds: List of bond dimensions, one for each layer in the tree. The first
      entry is the "physical dimension".
    new_top_rank: The top dimension of the tree. A value of 1 produces a pure
      state. A value > 1 produces an equal mixture of normalized pure states.

  Returns:
    isos_new: List of isometries defining the expanded tree.
  """
  old_Ds = [iso.shape[1] for iso in isos] + [isos[-1].shape[0]]

  if new_top_rank is None:
    new_top_rank = old_Ds[-1]
  new_Ds = new_Ds + [new_top_rank]

  if new_Ds[0] != old_Ds[0]:
    raise ValueError("Bottom dimension expansion not supported!")

  isos_new = [iso for iso in isos]
  for i in range(len(isos)):
    # Absorb dimension-expanding isometries on indices as needed
    if old_Ds[i + 1] != new_Ds[i + 1]:
      v = random_isometry(old_Ds[i + 1], new_Ds[i + 1], isos_new[i].dtype)
      isos_new[i] = tensornetwork.ncon([v, isos_new[i]], [(1, -1), (1, -2, -3)])
      if i + 1 < len(isos):
        isos_new[i + 1] = tensornetwork.ncon(
            [conj(v), conj(v), isos_new[i + 1]], [(1, -2), (2, -3), (-1, 1, 2)])
  return isos_new


def random_herm(D, dtype):
  """Generate a random hermitian matrix of dimension D.

  Symmetrizes a random matrix with entries drawn from a normal distribution.

  Args:
    D: The dimension.
    dtype: Element type.

  Returns:
    A random hermitian matrix.
  """
  h = random_normal_mat(D, D, dtype)
  return 0.5 * (h + adjoint(h))


def check_iso(iso):
  """Test the isometry property of a tree tensor network tensor.

  Args:
    iso: The supposed isometry.

  Returns:
    The norm difference between the square of iso and the identity.
  """
  sq = tensornetwork.ncon([iso, conj(iso)], [(-1, 1, 2), (-2, 1, 2)])
  return norm(sq - eye(sq.shape[0], dtype=sq.dtype))


def shift_ham(H, shift="auto"):
  """Add an identity contribution to a Hamiltonian.

  H -> H - shift * I

  Args:
    H: The local Hamiltonian term (2-tuple of 1-site contributions and 2-site
      MPO).
    shift: Amount by which to shift the spectrum downwards. If "auto", computes
      the spectrum of the local term H and shifts so that all eigenvalues are
      less than or equal to 0.

  Returns:
    The shifted Hamiltonian.
  """
  h1, (h2L, h2R) = H
  D = h1.shape[0]
  dtype = h1.dtype

  if shift == "auto":
    e1 = reduce_max(to_real(eigvalsh(h1)))

    h2 = sum([
        tensornetwork.ncon([hl, hr], [(-1, -3), (-2, -4)])
        for (hl, hr) in zip(h2L, h2R)
    ])
    h2 = reshape(h2, (D**2, D**2))
    e2 = reduce_max(to_real(eigvalsh(h2)))

    shift = cast(e1 + e2, dtype)

  if shift != 0.0:
    H = (h1 - shift * eye(D, dtype=dtype), (h2L, h2R))

  return H, shift


def _full_ham_top(H):
  """Compute the full Hamiltonian for the layer below the top tensor.

  Assuming periodic boundary conditions.

  Args:
    H: Local Hamiltonian ascended to just below the top tensor.

  Return:
    The full Hamiltonian for that layer as a dense matrix.
  """
  h1, (h2L, h2R) = H
  D = h1.shape[0]
  dtype = h1.dtype

  E = eye(D, dtype=dtype)

  fullH = tensornetwork.ncon([h1, E], [(-1, -3), (-2, -4)])
  fullH += tensornetwork.ncon([E, h1], [(-1, -3), (-2, -4)])
  for (hl, hr) in zip(h2L, h2R):
    fullH += tensornetwork.ncon([hl, hr], [(-1, -3), (-2, -4)])
  for (hl, hr) in zip(h2R, h2L):
    fullH += tensornetwork.ncon([hl, hr], [(-1, -3), (-2, -4)])

  return reshape(fullH, (D**2, D**2))


def _dense_ham_term(H):
  """Convert the dense representation of the local Hamiltonian term.

  Args:
    H: The sparse form for the local Hamiltonian term.

  Returns:
    The dense term as a single rank-4 tensor.
  """
  h1, (h2L, h2R) = H
  D = h1.shape[0]
  dtype = h1.dtype

  E = eye(D, dtype=dtype)

  h = tensornetwork.ncon([h1, E], [(-1, -3), (-2, -4)])
  for (hl, hr) in zip(h2L, h2R):
    h += tensornetwork.ncon([hl, hr], [(-1, -3), (-2, -4)])

  return h


def isos_with_transposes(isos_012):
  """Compute the transposes of all isometries in a tree.

  Args:
    isos_012: The isometries defining the tree.

  Returns:
    A list of tuples of form (iso_012, iso_021), with iso_021 the transpose
    (reflection) of iso_012.
  """
  return list(zip(isos_012, [transpose(w, (0, 2, 1)) for w in isos_012]))


def opt_tree_energy(isos_012,
                    H,
                    itr,
                    itr_l,
                    verbose=0,
                    graphed=False,
                    decomp_mode="svd_full_iso",
                    decomp_device=None,
                    envsq_dtype=None,
                    ham_shift="auto",
                    callback=None,
                    time_layer_updates=False):
  """Variationally minimize the energy of a binary tree tensor network.

  Spatial uniformity is assumed: The tree tensor network consists of a single
  isometric tensor per layer.

  The Hamiltonian, assumed to be translation invariant, is provided as a
  single nearest-neighbor term `H`. See for example `get_ham_ising()`, which
  constructs an appropriate object for the Ising model. The size of the
  second and third dimensions of the first-layer tensor `isos_012[0]` must
  match the physical dimension of the Hamiltonian.

  A number `itr` of variational sweeps are carried out. For each sweep, the
  tensor specifying each layer is optimized using a linear approximation,
  with `itr_l` iterations per layer.

  Args:
      isos_012: List of tensors specifying the tree tensor network; one
                tensor for each layer. Assumed to be isometries.
      H: The local term of the Hamiltonian as an MPO.
      itr: The number of variational sweeps to perform.
      itr_l: The number of iterations per layer. Typically, 1 is enough.
      verbose: Set to >0 to print some status information.
      graphed: If `True`, build a graph for a complete sweep for best
                performance.
      decomp_mode: Which decomposition scheme to use for tensor updates.
      decomp_device: TensorFlow device on which to perform decompositions.
      envsq_dtype: Data type to use for the squared environment. Only
                    applicable if `decomp_mode` is `"svd"` or `"eigh"`.
      ham_shift: Amount by which to shift the energies of the local 
                  Hamiltonian term. A small positive value typically improves
                  convergence.
      callback: A function to be called after each sweep.
      time_layer_updates: Boolean. Whether to collect timing data for layer
        updates, split into computation of environments and matrix
        decompositions. The data is supplied only to the callback function.
  Returns:
      isos_012: The optimized tensors of the tree tensor network.
  """
  with device(decomp_device):
    H, shift = shift_ham(H, ham_shift)
  print("Hamiltonian shift:", shift)

  L = len(isos_012)

  # Ascend through any trivial layers only once
  bottom = 0
  for l in range(L):
    shp = isos_012[l].shape
    if shp[0] == shp[1] * shp[2]:
      if graphed:
        H = ascend_uniform_op_local_graph(*H, isos_012[l],
                                  transpose(isos_012[l], (0, 2, 1)))
      else:
        H = ascend_uniform_op_local(*H, isos_012[l], transpose(
            isos_012[l], (0, 2, 1)))
      bottom = l + 1
    else:
      break

  t0 = time.time()
  for j in range(itr):
    if graphed:
      states = all_states_1site_graph(isos_012[bottom:])
    else:
      states = all_states_1site(isos_012[bottom:])
    states = [None] * bottom + states

    Hl = H
    svs = [None] * L
    tes_sweep = 0.0
    tds_sweep = 0.0
    for l in range(bottom, L):
      if verbose > 1:
        print("Optimizing level {}".format(l))
      res = opt_energy_layer(
          isos_012[l:],
          *Hl,
          states[l + 1:],
          itr_l,
          graphed=graphed,
          decomp_mode=decomp_mode,
          decomp_device=decomp_device,
          envsq_dtype=envsq_dtype,
          timing=time_layer_updates)
      isos_012[l], s = res[:2]
      svs[l] = s

      if time_layer_updates:
        tes, tds = res[2:]
        tes_sweep += tes
        tds_sweep += tds

      if l < L - 1:
        if graphed:
          Hl = ascend_uniform_op_local_graph(*Hl, isos_012[l],
                                     transpose(isos_012[l], (0, 2, 1)))
        else:
          Hl = ascend_uniform_op_local(*Hl, isos_012[l],
                               transpose(isos_012[l], (0, 2, 1)))

    if graphed:
      H_top = ascend_uniform_op_local_top_graph(*Hl, isos_012[-1],
                                        transpose(isos_012[-1], (0, 2, 1)))
    else:
      H_top = ascend_uniform_op_local_top(*Hl, isos_012[-1],
                                  transpose(isos_012[-1], (0, 2, 1)))
    en = trace(H_top) / (2**L) + shift * H_top.shape[0]

    tes_sweep = tes_sweep / (L + 1 - bottom)
    tds_sweep = tds_sweep / (L + 1 - bottom)

    if verbose > 0:
      minsv = np.min([to_numpy(sv).min() for sv in svs[bottom:]])
      print("sweeps: {}, energy density: {}, min_sv: {}, run-time: {}".format(
          j,
          to_numpy(en).real, minsv,
          time.time() - t0))

    if callback is not None:
      stop_request = callback(isos_012=isos_012,
                              decomp_singular_values=svs,
                              sweep_num=j,
                              energy=en,
                              time_sweep=time.time() - t0,
                              time_env=tes_sweep,
                              time_decomp=tds_sweep)
      if stop_request:
        break

  return isos_012


def top_hamiltonian(H, isos_012):
  """Ascend the Hamiltonian to the single leg on the top of the tree.

  In case the top rank is 1, this computes the Hamiltonian expectation value
  of the pure state.

  Args:
    H: The local Hamiltonian term for the bottom of the tree.
    isos_012: The isometries defining the tree.

  Returns:
    The Hamiltonian at the top, dimension equal to the top rank of the tree.
  """
  L = len(isos_012)
  for l in range(L - 1):
    H = ascend_uniform_op_local(
      *H, isos_012[l], transpose(isos_012[l], (0, 2, 1)))

  H = ascend_uniform_op_local_top(
    *H, isos_012[-1], transpose(isos_012[-1], (0, 2, 1)))
  
  return H


def top_eigen(H, isos_012):
  """Compute the eigenvalue decomposition of the top Hamiltonian.

  Args:
    H: The local Hamiltonian term for the bottom of the tree.
    isos_012: The isometries defining the tree.

  Returns:
    ev: Eigenvalues.
    eV: Matrix of eigenvectors.
  """
  Htop = top_hamiltonian(isos_012, H)
  return eigh(Htop)


def apply_top_op(isos_012, top_op):
  """Apply an operator to the top of a tree.

  Note: If the operator is not an isometry, the resulting tree will no longer
  be isometric.

  Args:
    isos_012: The isometries defining the tree.
    top_op: The operator to apply as a matrix. The right index will be 
      contracted with the top index of the tree.

  Returns:
    isos_012_new: Updated list of tensors defining the tree.
  """
  isos_012_new = isos_012[:]
  isos_012_new[-1] = tensornetwork.ncon(
    [top_op, isos_012_new[-1]],
    [(-1, 1), (1, -2, -3)])
  return isos_012_new


def apply_top_vec(isos_012, top_vec):
  """Contract a vector with the top of a tree, converting it to a pure state.

  Note: If the vector is not normalized, the tree will no longer be
  normalized, and hence no longer isometric.

  Args:
    isos_012: The isometries defining the tree.
    top_vec: Vector to contract with the tree top.

  Returns:
    isos_012_new: Updated list of tensors defining the tree.
  """
  if len(top_vec.shape) != 1:
    raise ValueError("top_purestate was not a vector!")
  top_op = reshape(top_vec, (1, top_vec.shape[0]))
  return apply_top_op(isos_012, top_op)


def top_translation(isos_012):
  """Ascend the physical translation operator to the top of the tree.

  For top rank equal to 1, this computes a value representing the translation
  invariance of the tree. If it is 1, the tree is completely translation
  invariant. Similarly, for top rank > 1, the unitarity of the resulting
  operator is a measure of the translation invariance of the mixed state.

  Args:
    isos_012: The isometries defining the tree.

  Returns:
    T: The coarse-grained translation operator.
  """
  d = isos_012[0].shape[1]
  E2 = eye(d**2, dtype=isos_012[0].dtype)
  # Ordering: mpo_left, mpo_right, phys_bottom, phys_top
  translation_tensor = reshape(E2, (d,d,d,d))
  return ascend_uniform_MPO_to_top(translation_tensor, isos_012)


def top_global_product_op(op, isos_012):
  """Ascend a uniform product of 1-site operators to the top of the tree.

  Args:
    op: 1-site operator (matrix) defining the global product.
    isos_012: The isometries defining the tree.

  Returns:
    top_op: The coarse-grained operator.
  """
  d = op.shape[0]
  Mop = reshape(op, (1, 1, d, d))
  return ascend_uniform_MPO_to_top(Mop, isos_012)


def top_localop_1site(op, n, isos_012):
  """Ascend a 1-site operator at a particular site to the top of the tree.

  Args:
    op: 1-site operator (matrix).
    n: The site number from which to ascend.
    isos_012: The isometries defining the tree.

  Returns:
    top_op: The coarse-grained operator.
  """
  L = len(isos_012)
  if not (0 <= n < 2**L):
    raise ValueError("Invalid site number '{}' with {} sites.".format(n, 2**L))
  for l in range(L):
    if n % 2 == 0:
      op = ascend_op_1site_to_1site_L(op, isos_012[l])
    else:
      op = ascend_op_1site_to_1site_R(op, isos_012[l])
    n = n // 2
  return op


def top_localop_2site(op, n, isos_012):
  """Ascend a 2-site MPO at a particular pair of sites to the top of the tree.

  Args:
    op: 2-site MPO (2-tuple of lists of operators).
    n: The (leftmost) site number from which to ascend.
    isos_012: The isometries defining the tree.

  Returns:
    top_op: The coarse-grained operator.
  """
  L = len(isos_012)
  N = 2**L
  if not (0 <= n < 2**L):
    raise ValueError("Invalid site number '{}' with {} sites.".format(n, N))
  np1 = n + 1  # site number of neighbor
  for l in range(L):
    xn = n // 2
    xnp1 = np1 // 2
    if n == np1:
      # After the ops merge, this is a 1-site op ascension.
      # Never occurs on the first iteration.
      if n % 2 == 0:
        op = ascend_op_1site_to_1site_L(op, isos_012[l])
      else:
        op = ascend_op_1site_to_1site_R(op, isos_012[l])
    elif (xn % 2 == 0) != (xnp1 % 2 == 0): 
      # If we are still following different paths
      if l == L-1: #catch the outside case
        op = ascend_op_2site_to_1site(
          reflect_mpo_2site(op), isos_012[l], transpose(isos_012[l], (0,2,1)))
      else:
        op = ascend_op_2site_to_2site(
          op, isos_012[l], transpose(isos_012[l], (0,2,1)))
    else:  # if the paths merge
      op = ascend_op_2site_to_1site(
        op, isos_012[l], transpose(isos_012[l], (0,2,1)))
    n = xn
    np1 = xnp1
  return op


def top_local_ham(H, n, isos_012):
  """Ascend a local Hamiltonian term at a particular location to the tree top.

  Keeps the 1-site and 2-site components separate.

  Args:
    H: Local Hamiltonian term in sparse representation.
    n: The (leftmost) site number from which to ascend.
    isos_012: The isometries defining the tree.

  Returns:
    top_op: The coarse-grained operator.
  """
  h1, h2 = H
  h1 = top_localop_1site(h1, n, isos_012)
  h2 = top_localop_2site(h2, n, isos_012)
  return (h1, h2)


def top_ham_all_terms(H, isos_012):
  """Ascend all Hamiltonian terms separately to the top of the tree.

  Args:
    H: Local Hamiltonian term in sparse representation.
    isos_012: The isometries defining the tree.

  Returns:
    top_ops: List of coarse-grained Hamiltonian terms.
  """
  N = 2**len(isos_012)
  Htop_terms = []
  for n in range(N):
    Htop_terms.append(top_local_ham(H, n, isos_012))
  return Htop_terms


def top_ham_modes(H, isos_012, ns):
  """Compute the Hamiltonian density modes at the top of the tree.

  Args:
    H: Local Hamiltonian term in sparse representation.
    isos_012: The isometries defining the tree.
    ns: Modes to compute (list of integers).

  Returns:
    mode_ops: List of coarse-grained Hamiltonian density modes.
  """
  Htop_terms = top_ham_all_terms(H, isos_012)
  N = len(Htop_terms)
  Hns = []
  for n in ns:
    Hn = sum(
      np.exp(1.j * n * j * 2*np.pi / N) * h1 + 
      np.exp(1.j * n * (j + 0.5) * 2*np.pi / N) * h2 
      for (j, (h1,h2)) in enumerate(Htop_terms))
    Hns.append(Hn)
  return Hns


def tree_energy_expval_check(H, isos_012):
  """Compute the energy at all levels in the tree.

  Useful for checking consistency of ascended Hamiltonians and descended
  states.

  Args:
    H: Local Hamiltonian term.
    isos_012: List of isometries defining the tree.

  Returns:
    Vector of energies, one for each level plus one for the top.
  """
  L = len(isos_012)
  states = all_states_1site(isos_012)

  ens = []
  Hl = H
  for l in range(L):
    en = _energy_expval_env(isos_012[l:], *Hl, states[l + 1:])
    ens.append(en / (2**L))
    if l < L - 1:
      Hl = ascend_uniform_op_local(*Hl, isos_012[l], transpose(
          isos_012[l], (0, 2, 1)))

  H_top = ascend_uniform_op_local_top(*Hl, isos_012[-1],
                              transpose(isos_012[-1], (0, 2, 1)))
  en = trace(H_top)
  ens.append(en / (2**L))

  return convert_to_tensor(ens)


def descend_full_state_pure(isos_012):
  """Compute the dense representation of the state from a pure tree.

  This is an expensive operation that requires exponential memory and time
  (in the number of sites, so doubly exponential in the number of layers!).

  Args:
    isos_012: The list of isometries defining the tree.

  Returns:
    The state as a dense tensor of rank N, where N is the number of sites.
  """
  if not isos_012[-1].shape[0] == 1:
    raise ValueError("Top dimension is not 1 (state not pure).")

  tree = tensornetwork.TensorNetwork()
  nisos = []

  iso_top = isos_012[-1]
  iso_top = reshape(iso_top, iso_top.shape[1:])

  niso = tree.add_node(
      iso_top,
      name="iso_{}_0".format(len(isos_012) - 1),
      axis_names=["bL", "bR"])
  nisos.append(niso)
  sites = [niso["bL"], niso["bR"]]

  for l in reversed(range(len(isos_012) - 1)):
    sites_next = []
    for (s, s_edge) in enumerate(sites):
      niso = tree.add_node(
          isos_012[l],
          name="iso_{}_{}".format(l, s),
          axis_names=["t", "bL", "bR"])
      tree.connect(s_edge, niso["t"])
      sites_next += [niso["bL"], niso["bR"]]
      nisos.append(niso)
    sites = sites_next

  nisos = nisos[::-1]
  nstate = nisos.pop()
  while nisos:
    nstate = tree.contract_between(nstate, nisos.pop())
  nstate = nstate.reorder_edges(sites)
  return nstate.get_tensor()


def get_ham_ising(dtype, J=1.0, h=1.0):
  """Return the local term for the critical Ising Hamiltonian.

  Defines the global Hamiltonian:
  $H = -\sum_{i=1}^N [ J * X_i X_{i+1} + h * Z_i ]$

  Args:
    dtype: The data type.
    J: The coupling strength.
    h: The field strength.

  Returns:
    The Hamiltonian term, separated into a 1-site contribution and a 2-site
    MPO.
  """
  X = convert_to_tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
  Z = convert_to_tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
  h_mpo_2site = ([-J * X], [X])
  h1 = -h * Z
  return h1, h_mpo_2site


def _weylops(q):
  om = np.exp(2j * np.pi / q)
  U = np.diag([om**j for j in range(q)])
  V = np.diag(np.ones(q - 1), 1)
  V[-1, 0] = 1
  return U, V, om


def get_ham_potts(dtype, q, J=1.0, h=1.0):
  """Return the local term for the q-state Potts Hamiltonian.

  Defines the global Hamiltonian:
  $H = -\sum_{i=1}^N \sum_{k=1}^q [ J * U_i^k U_{i+1}^{q-k} + h * V_i^k]$

  Args:
    dtype: The data type.
    q: Which root of unity to use. Alternatively, how many values the Potts
      "spins" are able to take.
    J: Coefficient for the nearest-neighbor terms (positive means
      ferromagnetic).
    h: Coefficient for the 1-site terms.

  Returns:
    The Hamiltonian term, separated into a 1-site contribution and a 2-site
    MPO.
  """
  U, V, _ = _weylops(q)

  mp = np.linalg.matrix_power

  if dtype_is_complex(dtype):
    h2 = ([-J * mp(U, k) for k in range(1, q)],
          [mp(U, q - k) for k in range(1, q)])
  else:
    # The straightforward way to build the Hamiltonian results in complex
    # matrices in the MPO. The dense Hamiltonian is, however, real.
    # To make the MPO real, we first build the dense 2-site term, then
    # use an SVD to split it back into a real MPO.
    h2_dense = sum(np.tensordot(-J * mp(U, k), mp(U, q-k), axes=((),()))
                for k in range(1,q))
    realness = np.linalg.norm(h2_dense - h2_dense.real)
    if realness > 1e-12:
      raise ValueError(
        "2-site term was not real. Realness = {}".format(realness))
    u, s, vh = svd_np(h2_dense.real.reshape((q**2, q**2)), full_matrices=False)
    mpo_rank = np.count_nonzero(s.round(decimals=12))
    if mpo_rank != q - 1:
      raise ValueError(
        "Error performing SVD of 2-site term. {} != {}-1".format(mpo_rank, q))
    h2 = ([s[i] * u[:,i].reshape(q,q) for i in range(q-1)],
          [vh[i,:].reshape(q,q) for i in range(q-1)])

  h1 = -h * sum(mp(V, k) for k in range(1, q))

  h1 = convert_to_tensor(h1, dtype=dtype)
  h2 = (
      [convert_to_tensor(h, dtype=dtype) for h in h2[0]],
      [convert_to_tensor(h, dtype=dtype) for h in h2[1]],
  )

  return h1, h2


def get_ham_heis_su3_2box(dtype):
  """Return the local term for the su(3) Heisenberg model at 2-box level.

  VERY EXPERIMENTAL: Requires su3 irrep matrices!

  Args:
    dtype: The data type.

  Returns:
    The Hamiltonian term, separated into a 1-site contribution and a 2-site
    MPO.
  """
  import scipy.io as sio
  su3_20 = sio.loadmat("experiments/tree_tensor_network/su3_20.mat")
  h2_dense = sum(np.tensordot(S, S, axes=((),()))
              for (k, S) in su3_20.items())
  realness = np.linalg.norm(h2_dense - h2_dense.real)
  if realness > 1e-12:
    raise ValueError(
      "2-site term was not real. Realness = {}".format(realness))
  u, s, vh = svd_np(h2_dense.real.reshape((6**2, 6**2)), full_matrices=False)
  mpo_rank = np.count_nonzero(s.round(decimals=12))
  if mpo_rank != 8:
    raise ValueError(
      "Error performing SVD of 2-site term. {} != {}".format(mpo_rank, 8))
  h2 = ([s[i] * u[:,i].reshape(6,6) for i in range(8)],
        [vh[i,:].reshape(6,6) for i in range(8)])
  h2 = (
      [convert_to_tensor(h, dtype=dtype) for h in h2[0]],
      [convert_to_tensor(h, dtype=dtype) for h in h2[1]],
  )
  h1 = zeros_like(h2[0][0])
  return h1, h2


def kron_td(a, b):
  """Computes the Kronecker product of two matrices using tensordot.

  Args:
    a: Matrix a.
    b: Matrix b.

  Returns:
    The Kronecker product a x b, as a matrix.
  """
  if len(a.shape) != 2 or len(b.shape) != 2:
    raise ValueError("Only implemented for matrices.")
  ab = tensordot(a, b, 0)
  ab = transpose(ab, (0,2,1,3))
  return reshape(ab, (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]))


def block_ham(H, sites_per_block):
  """Creates a 'blocked' Hamiltonian from an input Hamiltonian.

  Blocks sites together, increasing the site dimension.

  Args:
    H: The local Hamiltonian term.
    sites_per_block: The number of sites to block into one.

  Returns:
    The blocked local Hamiltonian term.
  """
  h1, h2 = H
  d = h1.shape[0]
  dtype = h1.dtype
  E = eye(d, dtype=dtype)

  h1_blk = None
  for i in range(sites_per_block):
    h1_term = h1 if i == 0 else E
    for j in range(1, sites_per_block):
      h1_term = kron_td(h1_term, h1 if i == j else E)
    if h1_blk is not None:
      h1_blk += h1_term
    else:
      h1_blk = h1_term

  h2_dense = sum(kron_td(h2[0][i], h2[1][i]) for i in range(len(h2[0])))
  for i in range(sites_per_block - 1):
    h1_term = h2_dense if i == 0 else E
    j = 2 if i == 0 else 1
    while j < sites_per_block:
      h1_term = kron_td(h1_term, h2_dense if i == j else E)
      j += 2 if i == j else 1
    h1_blk += h1_term
  del(h2_dense)

  E_big = eye(d**(sites_per_block - 1), dtype=dtype)
  h2_0 = [kron_td(E_big, h) for h in h2[0]]
  h2_1 = [kron_td(h, E_big) for h in h2[1]]

  return h1_blk, (h2_0, h2_1)


def get_ham_ising_tube(dtype, Ly, lam=-3.044):
  """Return the local term for the 2+1D Ising Hamiltonian on a narrow torus.

  Defines the global Hamiltonian:
  $H = -\sum_{\langle i, j \rangle} X_i X_j + lam * \sum_i Z_i ]$

  Represents the Hamiltonian for the 2D torus as a 1-dimensional Hamiltonian,
  where each "site" is a slice of the torus in the "y" direction. The site
  dimension thus depends on the size of the system in the y direction.

  Args:
    dtype: The data type.
    Ly: The size of the torus in the y direction (number of sites).
    lam: The field strength.

  Returns:
    The Hamiltonian term, separated into a 1-site contribution and a 2-site
    MPO.
  """
  X = np.array([[0.0, 1.0], [1.0, 0.0]])
  Z = np.array([[1.0, 0.0], [0.0, -1.0]])

  Xcol = [
      np.kron(np.kron(np.eye(2**i), X), np.eye(2**(Ly - i - 1)))
      for i in range(Ly)
  ]
  Zcol = [
      np.kron(np.kron(np.eye(2**i), Z), np.eye(2**(Ly - i - 1)))
      for i in range(Ly)
  ]

  Xcol = [convert_to_tensor(Xc, dtype=dtype) for Xc in Xcol]
  Zcol = [convert_to_tensor(Zc, dtype=dtype) for Zc in Zcol]

  h1 = lam * sum(Zcol) - sum(Xcol[i] @ Xcol[(i + 1) % Ly] for i in range(Ly))

  h_mpo_2site = ([-Xc for Xc in Xcol], Xcol)

  return h1, h_mpo_2site


def set_backend(backend):
  """Sets the backend to use for tree tensor network computations.

  A backend must be set after importing the module.

  Args:
    backend: Possible values are "tensorflow", "jax", and "numpy".
  """
  tensornetwork.set_default_backend(backend)

  global np
  global dtype_is_complex
  global random_normal_mat
  global conj
  global adjoint
  global build
  global trace
  global transpose
  global reshape
  global convert_to_tensor
  global device
  global cast
  global zeros_like
  global where
  global reduce_max
  global to_real
  global eye
  global diag
  global diag_part
  global sqrt
  global abvals
  global matmul
  global tensordot
  global norm
  global svd
  global svd_np
  global eigh
  global eigvalsh
  global qr
  global to_numpy
  global executing_eagerly

  if backend == "tensorflow":
    import numpy as np
    import scipy.linalg as spla
    import tensorflow as tf
    def dtype_is_complex(dtype):
      return dtype.is_complex
    def random_normal_mat(D1, D2, dtype):
      if dtype.is_complex:
        A = tf.complex(
          tf.random_normal((D1, D2), dtype=dtype.real_dtype),
          tf.random_normal((D1, D2), dtype=dtype.real_dtype)) / math.sqrt(2)
      else:
        A = tf.random_normal((D1, D2), dtype=dtype)
      return A
    conj = tf.math.conj
    adjoint = tf.linalg.adjoint
    def build(f):
      return tf.contrib.eager.defun(f, autograph=False)
    trace = tf.linalg.trace
    transpose = tf.transpose
    reshape = tf.reshape
    convert_to_tensor = tf.convert_to_tensor
    device = tf.device
    cast = tf.cast
    zeros_like = tf.zeros_like
    where = tf.where
    reduce_max = tf.reduce_max
    def to_real(x):
      return tf.cast(x, x.dtype.real_dtype)
    eye = tf.eye
    diag = tf.linalg.diag
    diag_part = tf.linalg.diag_part
    sqrt = tf.sqrt
    abvals = tf.abs
    matmul = tf.matmul
    tensordot = tf.tensordot
    norm = tf.norm
    svd = tf.linalg.svd
    svd_np = spla.svd
    eigh = tf.linalg.eigh
    eigvalsh = tf.linalg.eigvalsh
    qr = tf.linalg.qr
    def to_numpy(x):
      return x.numpy()
    executing_eagerly = tf.executing_eagerly
  elif backend in ("numpy", "jax"):
    if backend == "numpy":
      import numpy as np
      np_nojax = np
      import scipy.linalg as spla
    else:
      import numpy as np_nojax
      import jax.numpy as np
      import scipy.linalg as spla
    import contextlib
    def dtype_is_complex(dtype):
      return np_nojax.dtype(dtype).kind == 'c'
    def random_normal_mat(D1, D2, dtype):
      if dtype_is_complex(dtype):
        A = (np_nojax.random.randn(D1,D2) +
             1.j * np_nojax.random.randn(D1,D2)) / math.sqrt(2)
        A = np.asarray(A, dtype)
      else:
        A = np.asarray(np_nojax.random.randn(D1,D2), dtype)
      return A
    conj = np.conj
    def adjoint(x):
      return np.conj(np.transpose(x))
    if backend == "jax":
      from jax import jit
      build = jit
    else: 
      def build(x):
        return x
    trace = np.trace
    transpose = np.transpose
    reshape = np.reshape
    convert_to_tensor = np.array
    def device(_):
      return contextlib.suppress()  # a dummy context
    cast = np.asarray
    zeros_like = np.zeros_like
    where = np.where
    reduce_max = np.amax
    to_real = np.real
    eye = np.eye
    diag = np.diag
    diag_part = np.diagonal
    sqrt = np.sqrt
    abvals = np.abs
    def matmul(a, b, adjoint_b=False):
      if adjoint_b:
        return np.matmul(a, adjoint(b))
      return np.matmul(a, b)
    tensordot = np.tensordot
    norm = np.linalg.norm
    def svd(x):
      u, s, vh = np.linalg.svd(x, full_matrices=False)
      return s, u, adjoint(vh)
    svd_np = spla.svd
    eigh = np.linalg.eigh
    eigvalsh = np.linalg.eigvalsh
    qr = np.linalg.qr
    def to_numpy(x):
      return np.asarray(x)
    executing_eagerly = lambda: False
  else:
    raise ValueError("Unsupported backend: {}".format(backend))

  global ascend_uniform_op_local_graph
  global ascend_uniform_op_local_top_graph
  global opt_energy_layer_once_graph
  global _iso_from_envsq_decomp_graph
  global opt_energy_env_graph
  global _iso_from_svd_graph
  global _iso_from_svd_decomp_graph
  global all_states_1site_graph
  ascend_uniform_op_local_graph = build(ascend_uniform_op_local)
  ascend_uniform_op_local_top_graph = build(ascend_uniform_op_local_top)
  opt_energy_layer_once_graph = build(opt_energy_layer_once)
  _iso_from_envsq_decomp_graph = build(_iso_from_envsq_decomp)
  opt_energy_env_graph = build(opt_energy_env)
  _iso_from_svd_graph = build(_iso_from_svd)
  _iso_from_svd_decomp_graph = build(_iso_from_svd_decomp)
  all_states_1site_graph = build(all_states_1site)
