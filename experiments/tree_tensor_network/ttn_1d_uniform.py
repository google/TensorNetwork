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
"""1-dimensional uniform tree tensor network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import math
import time
import numpy as np
import scipy.linalg as spla

import tensorflow as tf

# FIXME: Hack to pull in tensornetwork from the parent directory.
sys.path.insert(0, "../../")
import tensornetwork
import ncon


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


def _ascend_partial(op, iso):
    """Ascend an operator through the right index of an isometry.
    For 012 (021) index ordering, this is equivalent to ascending from the
    physical right (left).
    Cost: D^4."""
    return ncon.ncon([iso, op], [(-1, -2, 0), (-3, 0)])


def _complete_partial_ascend(iso_op, iso):
    """Complete a partial operator ascension performed by `_ascend_partial()`.
    This contracts with the conjugated isometry.
    Cost: D^4."""
    return ncon.ncon([tf.conj(iso), iso_op], [(-1, 0, 1), (-2, 0, 1)])


def _ascend_op_2site_to_1site_partial(mpo_2site, iso_012, iso_021):
    op2L, op2R = mpo_2site

    M = len(op2L)  # MPO bond dimension

    terms = []
    for m in range(M):
        # permute result to 012 order: M mild transposes
        iso_op_mpo_L_012 = ncon.ncon([iso_021, op2L[m]], [(-1, -3, 0), (-2, 0)])

        terms.append(_ascend_partial(op2R[m], iso_op_mpo_L_012))
    iso_op_2site_012 = sum(terms)

    return iso_op_2site_012


def _ascend_op_1site_to_1site_partial(op_1site, iso_012, iso_021):
    iso_op_1site_L_021 = _ascend_partial(op_1site, iso_021)
    iso_op_1site_R_012 = _ascend_partial(op_1site, iso_012)
    return iso_op_1site_R_012, iso_op_1site_L_021


def _ascend_op_to_1site_partial(op_1site, mpo_2site, iso_012, iso_021):
    iso_op_2site_012 = _ascend_op_2site_to_1site_partial(
        mpo_2site, iso_012, iso_021)
    iso_op_1site_R_012, iso_op_1site_L_021 = _ascend_op_1site_to_1site_partial(
        op_1site, iso_012, iso_021)
    return iso_op_2site_012 + iso_op_1site_R_012, iso_op_1site_L_021


def ascend_op_1site_to_1site(op_1site, iso_012, iso_021):
    iso_op_1site_R_012, iso_op_1site_L_021 = _ascend_op_1site_to_1site_partial(
        op_1site, iso_012, iso_021)

    res = _complete_partial_ascend(iso_op_1site_L_021, iso_021)
    res += _complete_partial_ascend(iso_op_1site_R_012, iso_012)

    return res


def ascend_op_to_1site(op_1site, mpo_2site, iso_012, iso_021):
    terms_012, iso_op_1site_L_021 = _ascend_op_to_1site_partial(
        op_1site, mpo_2site, iso_012, iso_021)

    res = _complete_partial_ascend(iso_op_1site_L_021, iso_021)
    res += _complete_partial_ascend(terms_012, iso_012)

    return res


def ascend_op_2site_to_2site(mpo_2site, iso_012, iso_021):

    def _ascend(op, iso, iso_conj):
        return ncon.ncon([iso_conj, op, iso], [(-1, 2, 0), (0, 1), (-2, 2, 1)])

    op2L, op2R = mpo_2site
    dtype = iso_012.dtype

    M = len(op2L)

    iso_021_conj = tf.conj(iso_021)
    op_asc_R = []
    for m in range(M):
        op_asc_R.append(_ascend(op2R[m], iso_021, iso_021_conj))

    iso_012_conj = tf.conj(iso_012)
    op_asc_L = []
    for m in range(M):
        op_asc_L.append(_ascend(op2L[m], iso_012, iso_012_conj))

    return op_asc_L, op_asc_R


def ascend_op_local(op_1site, mpo_2site, iso_012, iso_021):
    op_1site = ascend_op_to_1site(op_1site, mpo_2site, iso_012, iso_021)
    mpo_2site = ascend_op_2site_to_2site(mpo_2site, iso_012, iso_021)
    return op_1site, mpo_2site


ascend_op_local_graph = tf.contrib.eager.defun(ascend_op_local)


def ascend_op_local_top(op_1site, mpo_2site, iso_012, iso_021):
    mpo_2site = add_mpos_2site(mpo_2site, reflect_mpo_2site(mpo_2site))
    op_1site = ascend_op_to_1site(op_1site, mpo_2site, iso_012, iso_021)
    return op_1site


ascend_op_local_top_graph = tf.contrib.eager.defun(ascend_op_local_top)


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
        mpo_tensor_dense = ncon.ncon([
            isos_012[l],
            tf.conj(isos_012[l]), mpo_tensor_dense, mpo_tensor_dense
        ], [(-4, 2, 0), (-3, 3, 4), (1, -2, 4, 0), (-1, 1, 3, 2)])
    op = ncon.ncon([mpo_tensor_dense], [(0, 0, -1, -2)])
    return op


def descend_state_1site_R(state_1site, iso_012):  #χ^4
    """Descends a state from the top to the rightmost index of the isometry `iso`.
    Physically, if `iso` has 012 ordering, this is a descent to the right and
    if `iso` has 021 ordering, this is a descent to the left.
    """
    return ncon.ncon(
        [iso_012, state_1site, tf.conj(iso_012)], [(1, 2, -1), (1, 0),
                                                   (0, 2, -2)])


def descend_state_1site_L(state_1site, iso_021):  #χ^4
    return descend_state_1site_R(state_1site, iso_021)


def descend_state_1site(state_1site, iso_012, iso_021):  #χ^4
    state_1L = descend_state_1site_L(state_1site, iso_021)
    state_1R = descend_state_1site_R(state_1site, iso_012)
    return 0.5 * (state_1L + state_1R)


def _descend_energy_env_L(env, iso_021):
    return [descend_state_1site_L(e, iso_021) for e in env]


def _descend_energy_env_R(env, iso_012):
    return [descend_state_1site_R(e, iso_012) for e in env]


def _mpo_with_state(iso_012, iso_021, h_mpo_2site, state_1site):
    # contract ascended hamiltonian at level `lup` with nearest 1-site descended state
    h2L, h2R = h_mpo_2site

    envL = [
        ncon.ncon(
            [state_1site, iso_021, h, tf.conj(iso_012)],
            [(0, 2), (0, -1, 1), (3, 1), (2, 3, -2)])  # one transpose required
        for h in h2L
    ]

    envR = [
        ncon.ncon(
            [state_1site, iso_012, h, tf.conj(iso_021)],
            [(0, 2), (0, -1, 1), (3, 1), (2, 3, -2)])  # one transpose required
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
        ncon.ncon([iso_021, h], [(-1, -3, 0), (-2, 0)]) for h in h2R
    ]  # transpose to 012
    iso_h2L_012 = [ncon.ncon([iso_012, h], [(-1, -2, 0), (-3, 0)]) for h in h2L]

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
            ncon.ncon([eL, ihR], [(0, -1), (0, -2, -3)])
            for eL, ihR in zip(envL, iso_h2_R))

        envR = sum(
            ncon.ncon([eR, ihL], [(0, -1), (0, -2, -3)])
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
    iso_021 = tf.transpose(iso_012, (0, 2, 1))
    terms_012, terms_021 = _ascend_op_to_1site_partial(h_op_1site, h_mpo_2site,
                                                       iso_012, iso_021)
    terms = terms_012 + tf.transpose(terms_021, (0, 2, 1))
    env = ncon.ncon([state_1site, terms], [(0, -1), (0, -2, -3)])
    return env


def opt_energy_env(isos_012,
                   h_op_1site,
                   h_mpo_2site,
                   states_1site_above,
                   envsq_dtype=None):
    if len(isos_012) == 1:  # top of tree
        h_mpo_2site = add_mpos_2site(h_mpo_2site,
                                     reflect_mpo_2site(h_mpo_2site))
        env = opt_energy_env_1site(isos_012[0], h_op_1site, h_mpo_2site,
                                   states_1site_above[0])
    else:
        env1 = opt_energy_env_1site(isos_012[0], h_op_1site, h_mpo_2site,
                                    states_1site_above[0])
        env2 = opt_energy_env_2site(isos_012, h_mpo_2site,
                                    states_1site_above[1:])
        env = env1 + env2

    if envsq_dtype is not None:
        env = tf.cast(env, envsq_dtype)

    env_sq = ncon.ncon([env, tf.conj(env)], [(-1, 0, 1), (-2, 0, 1)])
    return env, env_sq


def _uinv_decomp(X_sq, cutoff=0.0, decomp_mode="eigh", decomp_device=None):
    with tf.device(decomp_device):
        if decomp_mode == "svd":
            # hermitian, positive matrix, so eigvals = singular values
            e, v, _ = tf.svd(X_sq)
        elif decomp_mode == "eigh":
            e, v = tf.linalg.eigh(X_sq)
            e = tf.cast(
                e, e.dtype.real_dtype)  # The values here should be real anyway
        else:
            raise ValueError("Invalid decomp_mode: {}".format(decomp_mode))
        #print(e.numpy())

        # NOTE: Negative values are always due to precision problems.
        # NOTE: Inaccuracies here mean the final tensor is not exactly isometric!
        e_pinvsqrt = tf.where(e <= cutoff, tf.zeros_like(e), 1 / tf.sqrt(e))

        e_pinvsqrt_mat = tf.diag(tf.cast(e_pinvsqrt, v.dtype))
        X_uinv = tf.matmul(v @ e_pinvsqrt_mat, v, adjoint_b=True)
    return X_uinv, e


_uinv_decomp_graph = tf.contrib.eager.defun(_uinv_decomp)


def _energy_expval_env(isos_012, h_op_1site, h_mpo_2site, states_1site_above):
    if len(isos_012) == 1:  # top of tree
        h_mpo_2site = add_mpos_2site(h_mpo_2site,
                                     reflect_mpo_2site(h_mpo_2site))
        env = opt_energy_env_1site(isos_012[0], h_op_1site, h_mpo_2site,
                                   states_1site_above[0])
    else:
        env1 = opt_energy_env_1site(isos_012[0], h_op_1site, h_mpo_2site,
                                    states_1site_above[0])
        env2 = opt_energy_env_2site(isos_012, h_mpo_2site,
                                    states_1site_above[1:])
        env = env1 + env2 / 2
        # NOTE: There are *two* environments for each Ham. term spanning two
        #       isometries. To get the correct energy we must divide env2 by 2.
    nsites = 2**(len(isos_012) - 1)
    return ncon.ncon([tf.conj(isos_012[0]), env], [(0, 1, 2),
                                                   (0, 1, 2)]) * nsites


opt_energy_env_graph = tf.contrib.eager.defun(opt_energy_env)


def _iso_from_svd(u, vh):
    return ncon.ncon([u, vh], [(-1, 0), (0, -2, -3)])


_iso_from_svd_graph = tf.contrib.eager.defun(_iso_from_svd)


def _iso_from_uinv(env, env_uinv):
    return ncon.ncon([env_uinv, env], [(-1, 0), (0, -2, -3)])


_iso_from_uinv_graph = tf.contrib.eager.defun(_iso_from_uinv)


def _iso_from_svd_decomp(env, decomp_device=None):
    with tf.device(decomp_device):
        env_r = tf.reshape(env, (env.shape[0], -1))
        s, u, v = tf.svd(env_r)
        vh = tf.linalg.adjoint(v)
        vh = tf.reshape(vh, (vh.shape[0], env.shape[1], env.shape[2]))
        return u, s, vh


_iso_from_svd_decomp_graph = tf.contrib.eager.defun(_iso_from_svd_decomp)


def _iso_from_svd_decomp_scipy(env):
    env = env.numpy()
    env_r = env.reshape((env.shape[0], -1))
    u, s, vh = spla.svd(env_r, full_matrices=False)
    u = tf.convert_to_tensor(u)
    s = tf.convert_to_tensor(s)
    vh = vh.reshape((vh.shape[0], env.shape[1], env.shape[2]))
    vh = tf.convert_to_tensor(vh)
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
    t_env = time.time() - t0

    t0 = time.time()
    if decomp_mode == "svd_full_iso":
        if graphed:
            u, s, vh = _iso_from_svd_decomp_graph(
                env, decomp_device=decomp_device)
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
        iso_012_new = tf.cast(iso_012_new, dtype)
    t_decomp = time.time() - t0

    return iso_012_new, s, t_env, t_decomp


opt_energy_layer_once_graph = tf.contrib.eager.defun(opt_energy_layer_once)


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
    states = [tf.eye(isos_012[-1].shape[0], dtype=isos_012[0][0].dtype)]
    for l in reversed(range(len(isos_012))):
        iso_021 = tf.transpose(isos_012[l], (0, 2, 1))
        states.insert(0, descend_state_1site(states[0], isos_012[l], iso_021))
    return states


all_states_1site_graph = tf.contrib.eager.defun(all_states_1site)


def entanglement_specs_1site(isos_012):
    specs = []
    state = tf.eye(isos_012[-1].shape[0], dtype=isos_012[0][0].dtype)
    for l in reversed(range(len(isos_012))):
        iso_021 = tf.transpose(isos_012[l], (0, 2, 1))
        state = descend_state_1site(state, isos_012[l], iso_021)
        e = tf.linalg.eigvalsh(state)
        e = tf.cast(e, e.dtype.real_dtype)
        specs.insert(0, e)
    return specs


def entropies_from_specs(specs):
    entropies = []
    for (i, spec) in enumerate(specs):
        spec = spec.numpy()
        x = spec * np.log2(spec)
        x[np.isnan(x)] = 0.0  # ignore zero or negative eigenvalues
        S = -np.sum(x)
        entropies.append(S)
    return entropies


def random_isometry(D1, D2, dtype=tf.complex64):
    assert D1 <= D2
    if dtype.is_complex:
        A = tf.complex(
            tf.random_normal((D1, D2), dtype=dtype.real_dtype),
            tf.random_normal((D1, D2), dtype=dtype.real_dtype)) / math.sqrt(2)
    else:
        A = tf.random_normal((D1, D2), dtype=dtype)

    A_inv, _ = _uinv_decomp(tf.matmul(A, A, adjoint_b=True))

    return A_inv @ A


def random_tree_tn_uniform(Ds, dtype, top_rank=1):
    num_layers = len(Ds)
    Ds = Ds + [top_rank]
    isos = []
    for j in range(num_layers):
        if Ds[j + 1] == Ds[j]**2:
            iso = tf.eye(Ds[j + 1], dtype=dtype)
        else:
            iso = random_isometry(Ds[j + 1], Ds[j]**2, dtype)
        iso = tf.reshape(iso, (Ds[j + 1], Ds[j], Ds[j]))
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
            v = random_isometry(
                old_Ds[i + 1], new_Ds[i + 1], dtype=isos_new[i].dtype)
            isos_new[i] = ncon.ncon([v, isos_new[i]], [(0, -1), (0, -2, -3)])
            if i + 1 < len(isos):
                isos_new[i + 1] = ncon.ncon(
                    [tf.conj(v), tf.conj(v), isos_new[i + 1]], [(0, -2),
                                                                (1, -3),
                                                                (-1, 0, 1)])
    return isos_new


def random_herm(D, dtype):
    if dtype.is_complex:
        h = tf.complex(
            tf.random_normal((D, D), dtype=dtype.real_dtype),
            tf.random_normal((D, D), dtype=dtype.real_dtype))
    else:
        h = tf.random_normal((D, D), dtype=dtype)
    return 0.5 * (h + tf.linalg.adjoint(h))


def check_iso(iso):
    sq = ncon.ncon([iso, tf.conj(iso)], [(-1, 0, 1), (-2, 0, 1)])
    return tf.norm(sq - tf.eye(sq.shape[0], dtype=sq.dtype))


def shift_ham(H, shift="auto"):
    h1, (h2L, h2R) = H
    D = h1.shape[0]
    dtype = h1.dtype

    if shift == "auto":
        e1 = tf.reduce_max(tf.cast(tf.linalg.eigvalsh(h1), dtype.real_dtype))

        h2 = sum([
            ncon.ncon([hl, hr], [(-1, -3), (-2, -4)])
            for (hl, hr) in zip(h2L, h2R)
        ])
        h2 = tf.reshape(h2, (D**2, D**2))
        e2 = tf.reduce_max(tf.cast(tf.linalg.eigvalsh(h2), dtype.real_dtype))

        shift = tf.cast(e1 + e2, dtype)

    if shift != 0.0:
        H = (h1 - shift * tf.eye(D, dtype=dtype), (h2L, h2R))

    return H, shift


def _full_ham_top(H):
    h1, (h2L, h2R) = H
    D = h1.shape[0]
    dtype = h1.dtype

    E = tf.eye(D, dtype=dtype)

    fullH = ncon.ncon([h1, E], [(-1, -3), (-2, -4)])
    fullH += ncon.ncon([E, h1], [(-1, -3), (-2, -4)])
    for (hl, hr) in zip(h2L, h2R):
        fullH += ncon.ncon([hl, hr], [(-1, -3), (-2, -4)])
    for (hl, hr) in zip(h2R, h2L):
        fullH += ncon.ncon([hl, hr], [(-1, -3), (-2, -4)])

    return tf.reshape(fullH, (D**2, D**2))


def _dense_ham_term(H):
    h1, (h2L, h2R) = H
    D = h1.shape[0]
    dtype = h1.dtype

    E = tf.eye(D, dtype=dtype)

    h = ncon.ncon([h1, E], [(-1, -3), (-2, -4)])
    for (hl, hr) in zip(h2L, h2R):
        h += ncon.ncon([hl, hr], [(-1, -3), (-2, -4)])

    return h


def isos_with_transposes(isos_012):
    return list(zip(isos_012, [tf.transpose(w, (0, 2, 1)) for w in isos_012]))


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

    with tf.device(decomp_device):
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
                                          tf.transpose(isos_012[l], (0, 2, 1)))
            else:
                H = ascend_op_local(*H, isos_012[l],
                                    tf.transpose(isos_012[l], (0, 2, 1)))
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
                    Hl = ascend_op_local_graph(
                        *Hl, isos_012[l], tf.transpose(isos_012[l], (0, 2, 1)))
                else:
                    Hl = ascend_op_local(*Hl, isos_012[l],
                                         tf.transpose(isos_012[l], (0, 2, 1)))

        if graphed:
            H_top = ascend_op_local_top_graph(
                *Hl, isos_012[-1], tf.transpose(isos_012[-1], (0, 2, 1)))
        else:
            H_top = ascend_op_local_top(*Hl, isos_012[-1],
                                        tf.transpose(isos_012[-1], (0, 2, 1)))
        en = tf.trace(H_top) / (2**L) + shift * H_top.shape[0]

        tes_sweep = tes_sweep / (L + 1 - bottom)
        tds_sweep = tds_sweep / (L + 1 - bottom)

        if verbose > 0:
            minsv = np.min([sv.numpy().min() for sv in svs[bottom:]])
            print("sweeps: {}, energy density: {}, min_sv: {}, run-time: {}".
                  format(j,
                         en.numpy().real, minsv,
                         time.time() - t0))

        if callback is not None:
            stop_request = callback(isos_012, svs, j, en,
                                    time.time() - t0, tes_sweep, tds_sweep)
            if stop_request:
                break

    return isos_012


def tree_energy_expval_check(isos_012, H):
    L = len(isos_012)
    states = all_states_1site(isos_012)

    ens = []
    Hl = H
    for l in range(L):
        en = _energy_expval_env(isos_012[l:], *Hl, states[l + 1:])
        ens.append(en / (2**L))
        if l < L - 1:
            Hl = ascend_op_local(*Hl, isos_012[l],
                                 tf.transpose(isos_012[l], (0, 2, 1)))

    H_top = ascend_op_local_top(*Hl, isos_012[-1],
                                tf.transpose(isos_012[-1], (0, 2, 1)))
    en = tf.trace(H_top)
    ens.append(en / (2**L))

    return tf.convert_to_tensor(ens)


def descend_full_state_pure(isos_012):
    if not isos_012[-1].shape[0] == 1:
        raise ValueError("Top dimension is not 1 (state not pure).")

    tree = tensornetwork.TensorNetwork()
    nisos = []

    iso_top = isos_012[-1]
    iso_top = tf.reshape(iso_top, iso_top.shape[1:])

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
    X = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    Z = tf.convert_to_tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)
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

    h2 = [-J * mp(U, k) for k in range(1, q)], [
        mp(U, q - k) for k in range(1, q)
    ]
    h1 = -h * sum(mp(V, k) for k in range(1, q))

    h1 = tf.convert_to_tensor(h1, dtype=dtype)
    h2 = (
        [tf.convert_to_tensor(h, dtype=dtype) for h in h2[0]],
        [tf.convert_to_tensor(h, dtype=dtype) for h in h2[1]],
    )

    return h1, h2


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

    Xcol = [tf.convert_to_tensor(Xc, dtype=dtype) for Xc in Xcol]
    Zcol = [tf.convert_to_tensor(Zc, dtype=dtype) for Zc in Zcol]

    h1 = lam * sum(Zcol) - sum(Xcol[i] @ Xcol[(i + 1) % Ly] for i in range(Ly))

    h_mpo_2site = ([-Xc for Xc in Xcol], Xcol)

    return h1, h_mpo_2site
