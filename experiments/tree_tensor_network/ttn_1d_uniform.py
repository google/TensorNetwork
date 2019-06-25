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
"""1-dimensional uniform binary tree tensor network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import math
import time
import tensornetwork


"""
Index ordering conventions:

iso_012:

  0
  |
(iso)
 / \
1   2


iso_021:

  0
  |
(iso)
 / \
2   1
"""


def build(x):
  """Dummy function for building a python function into a graph."""
  return x


def _ascend_partial(op, iso):
  """Ascend an operator through the right index of an isometry.
    For 012 (021) index ordering, this is equivalent to ascending from the
    physical right (left).
    Cost: D^4."""
  return tensornetwork.ncon([iso, op], [(-1, -2, 1), (-3, 1)])


def _complete_partial_ascend(iso_op, iso):
  """Complete a partial operator ascension performed by `_ascend_partial()`.
    This contracts with the conjugated isometry.
    Cost: D^4."""
  return tensornetwork.ncon([conj(iso), iso_op], [(-1, 1, 2), (-2, 1, 2)])


def _ascend_op_2site_to_1site_partial(mpo_2site, iso_021):
  """Ascend a 2-site MPO operator through a single isometry.
    Produces a 1-site operator after completion via
    `_complete_partial_ascend()`.
    Cost: D^4."""
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


def _ascend_op_1site_to_1site_partial(op_1site, iso_012, iso_021):
  iso_op_1site_L_021 = _ascend_partial(op_1site, iso_021)
  iso_op_1site_R_012 = _ascend_partial(op_1site, iso_012)
  return iso_op_1site_R_012, iso_op_1site_L_021


def _ascend_op_to_1site_partial(op_1site, mpo_2site, iso_012, iso_021):
  iso_op_2site_012 = _ascend_op_2site_to_1site_partial(mpo_2site, iso_021)
  iso_op_1site_R_012, iso_op_1site_L_021 = _ascend_op_1site_to_1site_partial(
      op_1site, iso_012, iso_021)
  return iso_op_2site_012 + iso_op_1site_R_012, iso_op_1site_L_021


def ascend_op_1site_to_1site(op_1site, iso_012, iso_021):
  iso_op_1site_R_012, iso_op_1site_L_021 = _ascend_op_1site_to_1site_partial(
      op_1site, iso_012, iso_021)

  res = _complete_partial_ascend(iso_op_1site_L_021, iso_021)
  res += _complete_partial_ascend(iso_op_1site_R_012, iso_012)

  return res


def ascend_op_1site_to_1site_separate(op_1site, iso_012, iso_021):
  iso_op_1site_R_012, iso_op_1site_L_021 = _ascend_op_1site_to_1site_partial(
    op_1site, iso_012, iso_021)

  resL = _complete_partial_ascend(iso_op_1site_L_021, iso_021)
  resR = _complete_partial_ascend(iso_op_1site_R_012, iso_012)
  return resL, resR


def ascend_op_1site_to_1site_R(op_1site, iso_012):
  return _complete_partial_ascend(_ascend_partial(op_1site, iso_012), iso_012)


def ascend_op_1site_to_1site_L(op_1site, iso_012):
  iso_021 = transpose(iso_012, (0,2,1))
  return _complete_partial_ascend(_ascend_partial(op_1site, iso_021), iso_021)


def ascend_op_to_1site(op_1site, mpo_2site, iso_012, iso_021):
  terms_012, iso_op_1site_L_021 = _ascend_op_to_1site_partial(
      op_1site, mpo_2site, iso_012, iso_021)

  res = _complete_partial_ascend(iso_op_1site_L_021, iso_021)
  res += _complete_partial_ascend(terms_012, iso_012)

  return res


def ascend_op_2site_to_1site(mpo_2site, iso_012, iso_021):
  iso_op_2site_012 = _ascend_op_2site_to_1site_partial(mpo_2site, iso_021)
  return _complete_partial_ascend(iso_op_2site_012, iso_012)


def ascend_op_2site_to_2site(mpo_2site, iso_012, iso_021):

  def _ascend(op, iso, iso_conj):
    return tensornetwork.ncon([iso_conj, op, iso], [(-1, 3, 1), (1, 2),
                                                    (-2, 3, 2)])

  op2L, op2R = mpo_2site
  dtype = iso_012.dtype

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


def ascend_op_local(op_1site, mpo_2site, iso_012, iso_021):
  op_1site = ascend_op_to_1site(op_1site, mpo_2site, iso_012, iso_021)
  mpo_2site = ascend_op_2site_to_2site(mpo_2site, iso_012, iso_021)
  return op_1site, mpo_2site


ascend_op_local_graph = build(ascend_op_local)


def ascend_op_local_top(op_1site, mpo_2site, iso_012, iso_021):
  mpo_2site = add_mpos_2site(mpo_2site, reflect_mpo_2site(mpo_2site))
  op_1site = ascend_op_to_1site(op_1site, mpo_2site, iso_012, iso_021)
  return op_1site


ascend_op_local_top_graph = build(ascend_op_local_top)


def ascend_op_local_many(op_1site, mpo_2site, isos):
  ops = []
  for l in range(len(isos)):
    op_1site, mpo_2site = ascend_op_local(op_1site, mpo_2site, *isos[l])
    ops.append(op_1site, mpo_2site)
  return ops


def ascend_uniform_MPO_to_top(mpo_tensor_dense, isos_012):
  """MPO ordering:
          3
          |
       0--m--1
          |
          2
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


def descend_state_1site_R(state_1site, iso_012):  #χ^4
  """Descends a state from the top to the rightmost index of the isometry `iso`.
    Physically, if `iso` has 012 ordering, this is a descent to the right and
    if `iso` has 021 ordering, this is a descent to the left.
    """
  return tensornetwork.ncon(
      [iso_012, state_1site, conj(iso_012)], [(2, 3, -1), (2, 1),
                                                 (1, 3, -2)])


def descend_state_1site_L(state_1site, iso_021):  #χ^4
  return descend_state_1site_R(state_1site, iso_021)


def descend_state_1site(state_1site, iso_012, iso_021):  #χ^4
  state_1L = descend_state_1site_L(state_1site, iso_021)
  state_1R = descend_state_1site_R(state_1site, iso_012)
  return 0.5 * (state_1L + state_1R)


def correlations_2pt_1s(isos_012, op):
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
        asc_opL, asc_opR = ascend_op_1site_to_1site_separate(
          asc_op, iso_012, iso_021)
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


def _descend_energy_env_L(env, iso_021):
  return [descend_state_1site_L(e, iso_021) for e in env]


def _descend_energy_env_R(env, iso_012):
  return [descend_state_1site_R(e, iso_012) for e in env]


def _mpo_with_state(iso_012, iso_021, h_mpo_2site, state_1site):
  # contract ascended hamiltonian at level `lup` with nearest 1-site descended state
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


def reflect_mpo_2site(mpo_2site):
  return tuple(reversed(mpo_2site))


def add_mpos_2site(mpo1, mpo2):
  return (mpo1[0] + mpo2[0], mpo1[1] + mpo2[1])


def _ascend_op_2site_to_2site_many(mpo_2site, isos):
  ops = []
  for l in range(len(isos)):
    mpo_2site = ascend_op_2site_to_2site(mpo_2site, *isos[l])
    ops.append(mpo_2site)
  return ops


def opt_energy_env_2site(isos_012, h_mpo_2site, states_1site_above):
  isos_wt = isos_with_transposes(isos_012)
  iso_012, iso_021 = isos_wt[0]
  isos_wt_above = isos_wt[1:]
  levels_above = len(isos_wt_above)

  # Ascend two-site Hamiltonian terms through to the bottom of the final isometry
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
  iso_021 = transpose(iso_012, (0, 2, 1))
  terms_012, terms_021 = _ascend_op_to_1site_partial(h_op_1site, h_mpo_2site,
                                                     iso_012, iso_021)
  terms = terms_012 + transpose(terms_021, (0, 2, 1))
  env = tensornetwork.ncon([state_1site, terms], [(1, -1), (1, -2, -3)])
  return env


def opt_energy_env(isos_012,
                   h_op_1site,
                   h_mpo_2site,
                   states_1site_above,
                   envsq_dtype=None):
  if len(isos_012) == 1:  # top of tree
    h_mpo_2site = add_mpos_2site(h_mpo_2site, reflect_mpo_2site(h_mpo_2site))
    env = opt_energy_env_1site(isos_012[0], h_op_1site, h_mpo_2site,
                               states_1site_above[0])
  else:
    env1 = opt_energy_env_1site(isos_012[0], h_op_1site, h_mpo_2site,
                                states_1site_above[0])
    env2 = opt_energy_env_2site(isos_012, h_mpo_2site, states_1site_above[1:])
    env = env1 + env2

  if envsq_dtype is not None:
    env = cast(env, envsq_dtype)

  env_sq = tensornetwork.ncon([env, conj(env)], [(-1, 1, 2), (-2, 1, 2)])
  return env, env_sq


opt_energy_env_graph = build(opt_energy_env)


def _uinv_decomp(X_sq, cutoff=0.0, decomp_mode="eigh", decomp_device=None):
  with device(decomp_device):
    if decomp_mode == "svd":
      # hermitian, positive matrix, so eigvals = singular values
      e, v, _ = svd(X_sq)
    elif decomp_mode == "eigh":
      e, v = eigh(X_sq)
      e = to_real(e)  # The values here should be real anyway
    else:
      raise ValueError("Invalid decomp_mode: {}".format(decomp_mode))

    # NOTE: Negative values are always due to precision problems.
    # NOTE: Inaccuracies here mean the final tensor is not exactly isometric!
    e_pinvsqrt = where(e <= cutoff, zeros_like(e), 1 / sqrt(e))

    e_pinvsqrt_mat = diag(cast(e_pinvsqrt, v.dtype))
    X_uinv = matmul(v @ e_pinvsqrt_mat, v, adjoint_b=True)
  return X_uinv, e


_uinv_decomp_graph = build(_uinv_decomp)


def _energy_expval_env(isos_012, h_op_1site, h_mpo_2site, states_1site_above):
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


def _iso_from_uinv(env, env_uinv):
  return tensornetwork.ncon([env_uinv, env], [(-1, 1), (1, -2, -3)])


_iso_from_uinv_graph = build(_iso_from_uinv)


def _iso_from_svd_decomp(env, decomp_device=None):
  with device(decomp_device):
    env_r = reshape(env, (env.shape[0], -1))
    s, u, v = svd(env_r)
    vh = adjoint(v)
    vh = reshape(vh, (vh.shape[0], env.shape[1], env.shape[2]))
    return u, s, vh


_iso_from_svd_decomp_graph = build(_iso_from_svd_decomp)


def _iso_from_svd_decomp_scipy(env):
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
                          decomp_mode="eigh",
                          decomp_device=None,
                          envsq_dtype=None):
  dtype = isos_012[0].dtype

  t0 = time.time()
  if graphed:
    env, env_sq = opt_energy_env_graph(
        isos_012,
        h_op_1site,
        h_mpo_2site,
        states_1site_above,
        envsq_dtype=envsq_dtype)
  else:
    env, env_sq = opt_energy_env(
        isos_012,
        h_op_1site,
        h_mpo_2site,
        states_1site_above,
        envsq_dtype=envsq_dtype)

  if executing_eagerly():
    # FIXME: Hack to ensure values are ready
    test = to_numpy(env_sq[0,0])

  t_env = time.time() - t0

  t0 = time.time()
  if decomp_mode == "svd_full_iso":
    if graphed:
      u, s, vh = _iso_from_svd_decomp_graph(env, decomp_device=decomp_device)
      iso_012_new = _iso_from_svd_graph(u, vh)
    else:
      u, s, vh = _iso_from_svd_decomp(env, decomp_device=decomp_device)
      iso_012_new = _iso_from_svd(u, vh)
  elif decomp_mode == "svd_full_iso_scipy":
    u, s, vh = _iso_from_svd_decomp_scipy(env)
    if graphed:
      iso_012_new = _iso_from_svd_graph(u, vh)
    else:
      iso_012_new = _iso_from_svd(u, vh)
  else:
    if graphed:
      env_uinv, s = _uinv_decomp_graph(
          env_sq, decomp_mode=decomp_mode, decomp_device=decomp_device)
      iso_012_new = _iso_from_uinv_graph(env, env_uinv)
    else:
      env_uinv, s = _uinv_decomp(
          env_sq, decomp_mode=decomp_mode, decomp_device=decomp_device)
      iso_012_new = _iso_from_uinv(env, env_uinv)

  if envsq_dtype is not None:
    iso_012_new = cast(iso_012_new, dtype)

  if executing_eagerly():
    # FIXME: Hack to ensure values are ready
    test = to_numpy(iso_012_new[0,0,0])

  t_decomp = time.time() - t0

  return iso_012_new, s, t_env, t_decomp


opt_energy_layer_once_graph = build(opt_energy_layer_once)


def opt_energy_layer(isos_012,
                     h_op_1site,
                     h_mpo_2site,
                     states_1site_above,
                     itr,
                     graphed=False,
                     decomp_mode="eigh",
                     decomp_device=None,
                     envsq_dtype=None,
                     graph_level=None):
  shp = isos_012[0].shape
  if shp[0] == shp[1] * shp[2]:  # unitary, nothing to optimise
    return isos_012[0]

  iso_012 = isos_012[0]
  s = None
  tes, tds = 0.0, 0.0
  for _ in range(itr):
    if graph_level == "sweep":
      iso_012, s, _, _ = opt_energy_layer_once_graph(
          isos_012,
          h_op_1site,
          h_mpo_2site,
          states_1site_above,
          graphed=False,
          decomp_mode=decomp_mode,
          decomp_device=decomp_device,
          envsq_dtype=envsq_dtype)
    else:
      iso_012, s, te, td = opt_energy_layer_once(
          isos_012,
          h_op_1site,
          h_mpo_2site,
          states_1site_above,
          graphed=graphed,
          decomp_mode=decomp_mode,
          decomp_device=decomp_device,
          envsq_dtype=envsq_dtype)
      tes += te
      tds += td

  return iso_012, s, tes / itr, tds / itr


def all_states_1site(isos_012):
  states = [eye(isos_012[-1].shape[0], dtype=isos_012[0][0].dtype)]
  for l in reversed(range(len(isos_012))):
    iso_021 = transpose(isos_012[l], (0, 2, 1))
    states.insert(0, descend_state_1site(states[0], isos_012[l], iso_021))
  return states


all_states_1site_graph = build(all_states_1site)


def entanglement_specs_1site(isos_012):
  specs = []
  state = eye(isos_012[-1].shape[0], dtype=isos_012[0][0].dtype)
  for l in reversed(range(len(isos_012))):
    iso_021 = transpose(isos_012[l], (0, 2, 1))
    state = descend_state_1site(state, isos_012[l], iso_021)
    e = eigvalsh(state)
    e = to_real(e)
    specs.insert(0, e)
  return specs


def entropies_from_specs(specs):
  entropies = []
  for (i, spec) in enumerate(specs):
    spec = to_numpy(spec)
    x = spec * np.log2(spec)
    x[np.isnan(x)] = 0.0  # ignore zero or negative eigenvalues
    S = -np.sum(x)
    entropies.append(S)
  return entropies


def random_isometry(D1, D2, dtype):
  assert D1 <= D2
  A = random_normal_mat(D1, D2, dtype)
  A_inv, _ = _uinv_decomp(matmul(A, A, adjoint_b=True))
  return A_inv @ A


def random_tree_tn_uniform(Ds, dtype, top_rank=1):
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
  h = random_normal_mat(D, D, dtype)
  return 0.5 * (h + adjoint(h))


def check_iso(iso):
  sq = tensornetwork.ncon([iso, conj(iso)], [(-1, 1, 2), (-2, 1, 2)])
  return norm(sq - eye(sq.shape[0], dtype=sq.dtype))


def shift_ham(H, shift="auto"):
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
  h1, (h2L, h2R) = H
  D = h1.shape[0]
  dtype = h1.dtype

  E = eye(D, dtype=dtype)

  h = tensornetwork.ncon([h1, E], [(-1, -3), (-2, -4)])
  for (hl, hr) in zip(h2L, h2R):
    h += tensornetwork.ncon([hl, hr], [(-1, -3), (-2, -4)])

  return h


def isos_with_transposes(isos_012):
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
                    callback=None):
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
        callback: A function to be called after each sweep. Takes 7 arguments.
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
        H = ascend_op_local_graph(*H, isos_012[l],
                                  transpose(isos_012[l], (0, 2, 1)))
      else:
        H = ascend_op_local(*H, isos_012[l], transpose(
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
      isos_012[l], s, tes, tds = opt_energy_layer(
          isos_012[l:],
          *Hl,
          states[l + 1:],
          itr_l,
          graphed=graphed,
          decomp_mode=decomp_mode,
          decomp_device=decomp_device,
          envsq_dtype=envsq_dtype)
      svs[l] = s

      tes_sweep += tes
      tds_sweep += tds

      if l < L - 1:
        if graphed:
          Hl = ascend_op_local_graph(*Hl, isos_012[l],
                                     transpose(isos_012[l], (0, 2, 1)))
        else:
          Hl = ascend_op_local(*Hl, isos_012[l],
                               transpose(isos_012[l], (0, 2, 1)))

    if graphed:
      H_top = ascend_op_local_top_graph(*Hl, isos_012[-1],
                                        transpose(isos_012[-1], (0, 2, 1)))
    else:
      H_top = ascend_op_local_top(*Hl, isos_012[-1],
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
      stop_request = callback(isos_012, svs, j, en,
                              time.time() - t0, tes_sweep, tds_sweep)
      if stop_request:
        break

  return isos_012


def top_hamiltonian(isos_012, H):
  L = len(isos_012)
  for l in range(L - 1):
    H = ascend_op_local(
      *H, isos_012[l], transpose(isos_012[l], (0, 2, 1)))

  H = ascend_op_local_top(
    *H, isos_012[-1], transpose(isos_012[-1], (0, 2, 1)))
  
  return H


def top_eigen(isos_012, H):
  Htop = top_hamiltonian(isos_012, H)
  return eigh(Htop)


def project_tree(isos_012, top_iso):
  isos_012_proj = isos_012[:]
  isos_012_proj[-1] = tensornetwork.ncon(
    [top_iso, isos_012_proj[-1]],
    [(-1, 1), (1, -2, -3)])
  return isos_012_proj


def pure_tree(isos_012, top_purestate):
  if len(top_purestate.shape) != 1:
    raise ValueError("top_purestate was not a vector!")
  top_iso = reshape(top_purestate, (1, top_purestate.shape[0]))
  return project_tree(isos_012, top_iso)


def top_translation(isos_012):
  d = isos_012[0].shape[1]
  E2 = eye(d**2, dtype=isos_012[0].dtype)
  # Ordering: mpo_left, mpo_right, phys_bottom, phys_top
  translation_tensor = reshape(E2, (d,d,d,d))
  return ascend_uniform_MPO_to_top(translation_tensor, isos_012)


def top_global_product_op(op, isos_012):
  d = op.shape[0]
  Mop = reshape(op, (1, 1, d, d))
  return ascend_uniform_MPO_to_top(Mop, isos_012)


def top_localop_1site(op, n, isos_012):
  L = len(isos_012)
  for l in range(L):
    if n % 2 == 0:
      op = ascend_op_1site_to_1site_L(op, isos_012[l])
    else:
      op = ascend_op_1site_to_1site_R(op, isos_012[l])
    n = n // 2
  return op


def top_localop_2site(op, n, isos_012):
  L = len(isos_012)
  N = 2**L
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
  h1, h2 = H
  h1 = top_localop_1site(h1, n, isos_012)
  h2 = top_localop_2site(h2, n, isos_012)
  return (h1, h2)


def top_ham_all_terms(H, isos_012):
  N = 2**len(isos_012)
  Htop_terms = []
  for n in range(N):
    Htop_terms.append(top_local_ham(H, n, isos_012))
  return Htop_terms


def top_ham_modes(H, isos_012, ns):
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


def tree_energy_expval_check(isos_012, H):
  L = len(isos_012)
  states = all_states_1site(isos_012)

  ens = []
  Hl = H
  for l in range(L):
    en = _energy_expval_env(isos_012[l:], *Hl, states[l + 1:])
    ens.append(en / (2**L))
    if l < L - 1:
      Hl = ascend_op_local(*Hl, isos_012[l], transpose(
          isos_012[l], (0, 2, 1)))

  H_top = ascend_op_local_top(*Hl, isos_012[-1],
                              transpose(isos_012[-1], (0, 2, 1)))
  en = trace(H_top)
  ens.append(en / (2**L))

  return convert_to_tensor(ens)


def descend_full_state_pure(isos_012):
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

  nstate = nisos.pop(0)
  while len(nisos) > 0:
    nstate = tree.contract_between(nstate, nisos.pop(0))
  nstate = nstate.reorder_edges(sites)
  return nstate.get_tensor()


def get_ham_ising(dtype):
  X = convert_to_tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
  Z = convert_to_tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
  h_mpo_2site = ([-X], [X])
  h1 = -Z
  return h1, h_mpo_2site


def weylops(q):
  om = np.exp(2j * np.pi / q)
  U = np.diag([om**j for j in range(q)])
  V = np.diag(np.ones(q - 1), 1)
  V[-1, 0] = 1
  return U, V, om


def get_ham_potts(dtype, q, J=1.0, h=1.0):
  U, V, om = weylops(q)

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
  """Computes the Kronecker product of two matrices using tensordot."""
  if len(a.shape) != 2 or len(b.shape) != 2:
    raise ValueError("Only implemented for matrices.")
  ab = tensordot(a, b, 0)
  ab = transpose(ab, (0,2,1,3))
  return reshape(ab, (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]))


def block_ham(h1, h2, sites_per_block):
  """Creates a 'blocked' Hamiltonian from an input Hamiltonian."""
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
  tensornetwork.set_default_backend(backend)
  # TODO(amilsted): Do this differently. It's kind of awful!
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
  global sqrt
  global matmul
  global tensordot
  global norm
  global svd
  global svd_np
  global eigh
  global eigvalsh
  global to_numpy
  global executing_eagerly

  if backend == "tensorflow":
    import numpy as np
    import scipy.linalg as spla
    import tensorflow as tf
    def dtype_is_complex(dtype): return dtype.is_complex
    def random_normal_mat(D1, D2, dtype):
      if dtype.is_complex:
        A = tf.complex(
          tf.random_normal((D1, D2), dtype=dtype.real_dtype),
          tf.random_normal((D1, D2), dtype=dtype.real_dtype)) / math.sqrt(2)
      else:
        A = tf.random_normal((D1, D2), dtype=dtype)
      return A
    conj = tf.conj
    adjoint = tf.linalg.adjoint
    build = tf.contrib.eager.defun
    trace = tf.trace
    transpose = tf.transpose
    reshape = tf.reshape
    convert_to_tensor = tf.convert_to_tensor
    device = tf.device
    cast = tf.cast
    zeros_like = tf.zeros_like
    where = tf.where
    reduce_max = tf.reduce_max
    def to_real(x): return tf.cast(x, x.dtype.real_dtype)
    eye = tf.eye
    diag = tf.diag
    sqrt = tf.sqrt
    matmul = tf.matmul
    tensordot = tf.tensordot
    norm = tf.norm
    svd = tf.svd
    svd_np = spla.svd
    eigh = tf.linalg.eigh
    eigvalsh = tf.linalg.eigvalsh
    def to_numpy(x): return x.numpy()
    executing_eagerly = tf.executing_eagerly
  elif backend == "numpy" or backend == "jax":
    if backend == "numpy":
      import numpy as np
      np_nojax = np
      import scipy.linalg as spla
    else:
      import numpy as np_nojax
      import jax.numpy as np
      import scipy.linalg as spla
    import contextlib
    def dtype_is_complex(dtype): return np_nojax.dtype(dtype).kind == 'c'
    def random_normal_mat(D1, D2, dtype):
      if dtype_is_complex(dtype):
        A = (np_nojax.random.randn(D1,D2) +
             1.j * np_nojax.random.randn(D1,D2)) / math.sqrt(2)
        A = np.asarray(A, dtype)
      else:
        A = np.asarray(np_nojax.random.randn(D1,D2), dtype)
      return A
    conj = np.conj
    adjoint = lambda x: np.conj(np.transpose(x))
    if backend == "jax":
      from jax import jit
      build = jit
    else: 
      build = lambda x: x
    trace = np.trace
    transpose = np.transpose
    reshape = np.reshape
    convert_to_tensor = np.array
    device = lambda x: contextlib.suppress()
    cast = np.asarray
    zeros_like = np.zeros_like
    where = np.where
    reduce_max = np.amax
    to_real = np.real
    eye = np.eye
    diag = np.diag
    sqrt = np.sqrt
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
    def to_numpy(x): return np.asarray(x)
    executing_eagerly = lambda: False
  else:
    raise ValueError("Unsupported backend: {}".format(backend))

  global ascend_op_local_graph
  global ascend_op_local_top_graph
  global opt_energy_layer_once_graph
  global _uinv_decomp_graph
  global opt_energy_env_graph
  global _iso_from_svd_graph
  global _iso_from_uinv_graph
  global _iso_from_svd_decomp_graph
  global all_states_1site_graph
  ascend_op_local_graph = build(ascend_op_local)
  ascend_op_local_top_graph = build(ascend_op_local_top)
  opt_energy_layer_once_graph = build(opt_energy_layer_once)
  _uinv_decomp_graph = build(_uinv_decomp)
  opt_energy_env_graph = build(opt_energy_env)
  _iso_from_svd_graph = build(_iso_from_svd)
  _iso_from_uinv_graph = build(_iso_from_uinv)
  _iso_from_svd_decomp_graph = build(_iso_from_svd_decomp)
  all_states_1site_graph = build(all_states_1site)
