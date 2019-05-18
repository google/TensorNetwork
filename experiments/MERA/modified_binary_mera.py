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
modified binary MERA optimization
parts of the following code are based on code written by Glen Evenbly (c) for www.tensors.net, (v1.1) 
"""

import sys
sys.path.append('../')
NUM_THREADS = 4
import os
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
import tensorflow as tf
import copy
import numpy as np
import time
import pickle
import ncon as ncon
import misc_mera
from sys import stdout

config = tf.ConfigProto()
config.intra_op_parallelism_threads = NUM_THREADS
config.inter_op_parallelism_threads = NUM_THREADS
tf.enable_eager_execution(config=config)
tf.enable_v2_behavior()


@tf.contrib.eager.defun
def ascending_super_operator(hamAB, hamBA, w_isometry, v_isometry, unitary,
                             refsym):
  """
    ascending super operator for a modified binary MERA
    ascends 'hamAB' and 'hamBA' up one layer
    Parameters:
    -------------------------
    hamAB, hamBA:    tf.Tensor
                     local Hamiltonian terms
    w_isometry:      tf.Tensor
    v_isometry:      tf.Tensor
    unitary:         tf.Tensor
    refsym:          bool 
                     if true, enforce reflection symmetry
    Returns: 
    ------------------------
    (hamABout, hamBAout):  tf.Tensor, tf.Tensor

    """

  indList1 = [[6, 4, 1, 2], [1, 3, -3], [6, 7, -1], [2, 5, 3, 9], [4, 5, 7, 10],
              [8, 9, -4], [8, 10, -2]]
  indList2 = [[3, 4, 1, 2], [5, 6, -3], [5, 7, -1], [1, 2, 6, 9], [3, 4, 7, 10],
              [8, 9, -4], [8, 10, -2]]
  indList3 = [[5, 7, 2, 1], [8, 9, -3], [8, 10, -1], [4, 2, 9, 3],
              [4, 5, 10, 6], [1, 3, -4], [7, 6, -2]]
  indList4 = [[3, 6, 2, 5], [2, 1, -3], [3, 1, -1], [5, 4, -4], [6, 4, -2]]

  hamBAout = ncon.ncon([
      hamAB, w_isometry,
      tf.conj(w_isometry), unitary,
      tf.conj(unitary), v_isometry,
      tf.conj(v_isometry)
  ], indList1)
  if refsym:
    hamBAout = hamBAout + tf.transpose(hamBAout, (1, 0, 3, 2))
  else:
    hamBAout = hamBAout + ncon.ncon([
        hamAB, w_isometry,
        tf.conj(w_isometry), unitary,
        tf.conj(unitary), v_isometry,
        tf.conj(v_isometry)
    ], indList3)

  hamBAout = hamBAout + ncon.ncon([
      hamBA, w_isometry,
      tf.conj(w_isometry), unitary,
      tf.conj(unitary), v_isometry,
      tf.conj(v_isometry)
  ], indList2)
  hamABout = ncon.ncon(
      [hamBA, v_isometry,
       tf.conj(v_isometry), w_isometry,
       tf.conj(w_isometry)], indList4)

  return hamABout, hamBAout


@tf.contrib.eager.defun
def descending_super_operator(rhoAB, rhoBA, w_isometry, v_isometry, unitary,
                              refsym):
  """
    descending super operator for a modified binary MERA
    """

  indList1 = [[9, 3, 4, 2], [-3, 5, 4], [-1, 10, 9], [-4, 7, 5, 6],
              [-2, 7, 10, 8], [1, 6, 2], [1, 8, 3]]
  indList2 = [[3, 6, 2, 5], [1, 7, 2], [1, 9, 3], [-3, -4, 7, 8],
              [-1, -2, 9, 10], [4, 8, 5], [4, 10, 6]]
  indList3 = [[3, 9, 2, 4], [1, 5, 2], [1, 8, 3], [7, -3, 5, 6], [7, -1, 8, 10],
              [-4, 6, 4], [-2, 10, 9]]
  indList4 = [[3, 6, 2, 5], [-3, 1, 2], [-1, 1, 3], [-4, 4, 5], [-2, 4, 6]]

  rhoABout = 0.5 * ncon.ncon([
      rhoBA, w_isometry,
      tf.conj(w_isometry), unitary,
      tf.conj(unitary), v_isometry,
      tf.conj(v_isometry)
  ], indList1)

  if refsym:
    rhoABout = rhoABout + tf.transpose(rhoABout, (1, 0, 3, 2))
  else:
    rhoABout = rhoABout + 0.5 * ncon.ncon([
        rhoBA, w_isometry,
        tf.conj(w_isometry), unitary,
        tf.conj(unitary), v_isometry,
        tf.conj(v_isometry)
    ], indList3)

  rhoBAout = 0.5 * ncon.ncon([
      rhoBA, w_isometry,
      tf.conj(w_isometry), unitary,
      tf.conj(unitary), v_isometry,
      tf.conj(v_isometry)
  ], indList2)

  rhoBAout = rhoBAout + 0.5 * ncon.ncon(
      [rhoAB, v_isometry,
       tf.conj(v_isometry), w_isometry,
       tf.conj(w_isometry)], indList4)

  return rhoABout, rhoBAout


@tf.contrib.eager.defun
def get_env_disentangler(hamAB, hamBA, rhoBA, w, v, u, refsym):

  indList1 = [[7, 8, 10, -1], [4, 3, 9, 2], [10, -3, 9], [7, 5, 4],
              [8, -2, 5, 6], [1, -4, 2], [1, 6, 3]]
  indList2 = [[7, 8, -1, -2], [3, 6, 2, 5], [1, -3, 2], [1, 9, 3],
              [7, 8, 9, 10], [4, -4, 5], [4, 10, 6]]
  indList3 = [[7, 8, -2, 10], [3, 4, 2, 9], [1, -3, 2], [1, 5, 3],
              [-1, 7, 5, 6], [10, -4, 9], [8, 6, 4]]

  uEnv = ncon.ncon(
      [hamAB, rhoBA, w, tf.conj(w),
       tf.conj(u), v, tf.conj(v)], indList1)
  if refsym:
    uEnv = uEnv + tf.transpose(uEnv, (1, 0, 3, 2))
  else:
    uEnv = uEnv + ncon.ncon(
        [hamAB, rhoBA, w,
         tf.conj(w), tf.conj(u), v,
         tf.conj(v)], indList3)

  uEnv = uEnv + ncon.ncon(
      [hamBA, rhoBA, w, tf.conj(w),
       tf.conj(u), v, tf.conj(v)], indList2)

  return uEnv


@tf.contrib.eager.defun
def get_env_w_isometry(hamAB, hamBA, rhoBA, rhoAB, w_isometry, v_isometry,
                       unitary):
  """
    Parameters:
    """
  indList1 = [[7, 8, -1, 9], [4, 3, -3, 2], [7, 5, 4], [9, 10, -2, 11],
              [8, 10, 5, 6], [1, 11, 2], [1, 6, 3]]
  indList2 = [[1, 2, 3, 4], [10, 7, -3, 6], [-1, 11, 10], [3, 4, -2, 8],
              [1, 2, 11, 9], [5, 8, 6], [5, 9, 7]]
  indList3 = [[5, 7, 3, 1], [10, 9, -3, 8], [-1, 11, 10], [4, 3, -2, 2],
              [4, 5, 11, 6], [1, 2, 8], [7, 6, 9]]
  indList4 = [[3, 7, 2, -1], [5, 6, 4, -3], [2, 1, 4], [3, 1, 5], [7, -2, 6]]

  wEnv = ncon.ncon([
      hamAB, rhoBA,
      tf.conj(w_isometry), unitary,
      tf.conj(unitary), v_isometry,
      tf.conj(v_isometry)
  ], indList1)
  wEnv = wEnv + ncon.ncon([
      hamBA, rhoBA,
      tf.conj(w_isometry), unitary,
      tf.conj(unitary), v_isometry,
      tf.conj(v_isometry)
  ], indList2)

  wEnv = wEnv + ncon.ncon([
      hamAB, rhoBA,
      tf.conj(w_isometry), unitary,
      tf.conj(unitary), v_isometry,
      tf.conj(v_isometry)
  ], indList3)

  wEnv = wEnv + ncon.ncon(
      [hamBA, rhoAB, v_isometry,
       tf.conj(v_isometry),
       tf.conj(w_isometry)], indList4)

  return wEnv


@tf.contrib.eager.defun
def get_env_v_isometry(hamAB, hamBA, rhoBA, rhoAB, w_isometry, v_isometry,
                       unitary):

  indList1 = [[6, 4, 1, 3], [9, 11, 8, -3], [1, 2, 8], [6, 7, 9], [3, 5, 2, -2],
              [4, 5, 7, 10], [-1, 10, 11]]
  indList2 = [[3, 4, 1, 2], [8, 10, 9, -3], [5, 6, 9], [5, 7, 8], [1, 2, 6, -2],
              [3, 4, 7, 11], [-1, 11, 10]]
  indList3 = [[9, 10, 11, -1], [3, 4, 2, -3], [1, 8, 2], [1, 5, 3],
              [7, 11, 8, -2], [7, 9, 5, 6], [10, 6, 4]]
  indList4 = [[7, 5, -1, 4], [6, 3, -3, 2], [7, -2, 6], [4, 1, 2], [5, 1, 3]]

  vEnv = ncon.ncon([
      hamAB, rhoBA, w_isometry,
      tf.conj(w_isometry), unitary,
      tf.conj(unitary),
      tf.conj(v_isometry)
  ], indList1)
  vEnv = vEnv + ncon.ncon([
      hamBA, rhoBA, w_isometry,
      tf.conj(w_isometry), unitary,
      tf.conj(unitary),
      tf.conj(v_isometry)
  ], indList2)
  vEnv = vEnv + ncon.ncon([
      hamAB, rhoBA, w_isometry,
      tf.conj(w_isometry), unitary,
      tf.conj(unitary),
      tf.conj(v_isometry)
  ], indList3)
  vEnv = vEnv + ncon.ncon(
      [hamBA, rhoAB,
       tf.conj(v_isometry), w_isometry,
       tf.conj(w_isometry)], indList4)

  return vEnv


@tf.contrib.eager.defun
def steady_state_density_matrices(nsteps, rhoAB, rhoBA, w_isometry, v_isometry,
                                  unitary, refsym):
  for n in range(nsteps):
    rhoAB_, rhoBA_ = descending_super_operator(rhoAB, rhoBA, w_isometry,
                                               v_isometry, unitary, refsym)
    rhoAB = 1 / 2 * (rhoAB_ + tf.conj(tf.transpose(rhoAB_,
                                                   (2, 3, 0, 1)))) / ncon.ncon(
                                                       [rhoAB_], [[1, 2, 1, 2]])
    rhoBA = 1 / 2 * (rhoBA_ + tf.conj(tf.transpose(rhoBA_,
                                                   (2, 3, 0, 1)))) / ncon.ncon(
                                                       [rhoBA_], [[1, 2, 1, 2]])

  return rhoAB, rhoBA


#@tf.contrib.eager.defun  #better not defun this function, it takes ages to compile the graph
def optimize_mod_binary_mera(hamAB_0,
                             hamBA_0,
                             rhoAB_0,
                             rhoBA_0,
                             wC,
                             vC,
                             uC,
                             numiter=1000,
                             refsym=False,
                             nsteps_steady_state=4,
                             verbose=0,
                             opt_u=True,
                             opt_vw=True,
                             numpy_update_u=True):
  """
    ------------------------
    adapted from Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 24/1/2019
    ------------------------
    optimization of a scale invariant modified binary MERA tensor network
    Parameters:
    ----------------------------
    hamAB_0, hamBA_0:      tf.Tensor
                           bottom-layer Hamiltonians in AB and BA sublattices
    rhoAB_0, rhoBA_0:      tf.Tensor 
                           initial values for steady-state density matrices
    wC, vC, uC:            list of tf.Tensor 
                           isometries (wC, vC) and disentanglers (uC) of the MERA, with 
                           bottom layers first 
    numiter:               int 
                           number of iteration steps 
    refsym:                bool 
                           impose reflection symmetry 
    nsteps_steady_state:   int 
                           number of power-methodf iteration steps for calculating the 
                           steady state density matrices 
    verbose:               int 
                           verbosity flag 
    opt_u, opt_uv:         bool 
                           if False, skip unitary or isometry optimization 
    numpy_update_u:        bool
                           if True, use numpy svd to calculate update of disentanglers

    Returns: 
    -------------------------------
    (wC, vC, uC, rhoAB, rhoBA, run_times, Energies)
    wC, vC, uC:             list of tf.Tensor 
                            obtimized MERA tensors
    rhoAB, rhoBA:           tf.Tensor 
                            steady state density matrices at the top layer 
    run_times:              list 
                            run times per iteration step 
    Energies:               list 
                            energies at each iteration step
    """
  dtype = rhoAB_0.dtype

  hamAB = [0 for x in range(len(vC) + 1)]
  hamBA = [0 for x in range(len(vC) + 1)]
  rhoAB = [0 for x in range(len(vC) + 1)]
  rhoBA = [0 for x in range(len(vC) + 1)]

  hamAB[0] = hamAB_0
  hamBA[0] = hamBA_0

  chi1 = hamAB[0].shape[0]

  bias = tf.math.reduce_max(
      tf.linalg.eigvalsh(tf.reshape(hamAB[0], (chi1 * chi1, chi1 * chi1))))
  hamAB[0] = hamAB[0] - bias * tf.reshape(
      tf.eye(chi1 * chi1, dtype=dtype), (chi1, chi1, chi1, chi1))
  hamBA[0] = hamBA[0] - bias * tf.reshape(
      tf.eye(chi1 * chi1, dtype=dtype), (chi1, chi1, chi1, chi1))

  Energies = []
  run_times = []

  for k in range(numiter):
    t1 = time.time()
    rhoAB[-1], rhoBA[-1] = steady_state_density_matrices(
        nsteps_steady_state, rhoAB_0, rhoBA_0, wC[-1], vC[-1], uC[-1], refsym)
    for p in range(len(rhoAB) - 2, -1, -1):
      rhoAB[p], rhoBA[p] = descending_super_operator(
          rhoAB[p + 1], rhoBA[p + 1], wC[p], vC[p], uC[p], refsym)

    if verbose > 0:
      if np.mod(k, 10) == 1:
        Energies.append(
            (ncon.ncon([rhoAB[0], hamAB[0]], [[1, 2, 3, 4], [1, 2, 3, 4]]) +
             ncon.ncon([rhoBA[0], hamBA[0]], [[1, 2, 3, 4], [1, 2, 3, 4]])
            ) / 4 + bias / 2)
        stdout.write(
            '\rIteration: %i of %i: E = %.8f, err = %.16f at D = %i with %i layers'
            % (int(k), int(numiter), float(Energies[-1]),
               float(Energies[-1] + 4 / np.pi,), int(wC[-1].shape[2]), len(wC)))
        stdout.flush()

    for p in range(len(wC)):
      if opt_u and (k % opt_u == 0):
        uEnv = get_env_disentangler(hamAB[p], hamBA[p], rhoBA[p + 1], wC[p],
                                    vC[p], uC[p], refsym)
        if refsym:
          uEnv = uEnv + tf.transpose(uEnv, (1, 0, 3, 2))
        if numpy_update_u:
          uC[p] = misc_mera.u_update_svd_numpy(uEnv)
        else:
          uC[p] = misc_mera.u_update_svd(uEnv)

      if opt_vw:
        wEnv = get_env_w_isometry(hamAB[p], hamBA[p], rhoBA[p + 1],
                                  rhoAB[p + 1], wC[p], vC[p], uC[p])
        wC[p] = misc_mera.w_update_svd(wEnv)
        if refsym:
          vC[p] = wC[p]
        else:
          vEnv = get_env_v_isometry(hamAB[p], hamBA[p], rhoBA[p + 1],
                                    rhoAB[p + 1], wC[p], vC[p], uC[p])
          vC[p] = misc_mera.w_update_svd(vEnv)

      hamAB[p + 1], hamBA[p + 1] = ascending_super_operator(
          hamAB[p], hamBA[p], wC[p], vC[p], uC[p], refsym)

    run_times.append(time.time() - t1)
    if verbose > 2:
      print('time per iteration: ', run_times[-1])

  return wC, vC, uC, rhoAB[-1], rhoBA[-1], run_times, Energies


def increase_bond_dimension_by_adding_layers(chi_new, wC, vC, uC):
  """
    increase the bond dimension of the MERA to `chi_new`
    by padding tensors in the last layer with zeros. If the desired `chi_new` cannot
    be obtained from padding, adds layers of Tensors
    the last layer is guaranteed to have uniform bond dimension

    Parameters:
    --------------------------------
    chi_new:         int 
                     new bond dimenion
    wC, vC, uC:      list of tf.Tensor 
                     MERA isometries and disentanglers


    Returns:         
    --------------------------------
    (wC, vC, uC):    list of tf.Tensors
    """
  if misc_mera.all_same_chi(wC[-1], vC[-1],
                            uC[-1]) and (wC[-1].shape[2] >= chi_new):
    #nothing to do here
    return wC, vC, uC
  elif misc_mera.all_same_chi(wC[-1], vC[-1],
                              uC[-1]) and (wC[-1].shape[2] < chi_new):
    chi = min(chi_new, wC[-1].shape[0] * wC[-1].shape[1])
    wC[-1] = misc_mera.pad_tensor(wC[-1],
                                  [wC[-1].shape[0], wC[-1].shape[1], chi])
    vC[-1] = misc_mera.pad_tensor(vC[-1],
                                  [vC[-1].shape[0], vC[-1].shape[1], chi])
    wC_temp = copy.deepcopy(wC[-1])
    vC_temp = copy.deepcopy(vC[-1])
    uC_temp = copy.deepcopy(uC[-1])
    wC.append(misc_mera.pad_tensor(wC_temp, [chi, chi, chi]))
    vC.append(misc_mera.pad_tensor(vC_temp, [chi, chi, chi]))
    uC.append(misc_mera.pad_tensor(uC_temp, [chi, chi, chi, chi]))
    return increase_bond_dimension_by_adding_layers(chi_new, wC, vC, uC)

  elif not misc_mera.all_same_chi(wC[-1], vC[-1], uC[-1]):
    raise ValueError('chis of last layer have to be all the same!')


def increase_bond_dimension_by_padding(chi_new, wC, vC, uC):
  """
    increase the bond dimension of the MERA to `chi_new`
    by padding tensors in all layers with zeros. If the desired `chi_new` cannot
    be obtained from padding, adds layers of Tensors
    the last layer is guaranteed to have uniform bond dimension

    Parameters:
    --------------------------------
    chi_new:         int 
                     new bond dimenion
    wC, vC, uC:      list of tf.Tensor 
                     MERA isometries and disentanglers


    Returns: 
    --------------------------------
    (wC, vC, uC):    list of tf.Tensors
    """

  all_chis = [t.shape[n] for t in wC for n in range(len(t.shape))]
  if not np.all([c <= chi_new for c in all_chis]):
    #nothing to increase
    return wC, vC, uC

  chi_0 = wC[0].shape[0]
  wC[0] = misc_mera.pad_tensor(wC[0], [chi_0, chi_0, min(chi_new, chi_0**2)])
  vC[0] = misc_mera.pad_tensor(vC[0], [chi_0, chi_0, min(chi_new, chi_0**2)])

  for n in range(1, len(wC)):
    wC[n] = misc_mera.pad_tensor(wC[n], [
        min(chi_new, chi_0**(2 * n)),
        min(chi_new, chi_0**(2 * n)),
        min(chi_new, chi_0**(4 * n))
    ])
    vC[n] = misc_mera.pad_tensor(vC[n], [
        min(chi_new, chi_0**(2 * n)),
        min(chi_new, chi_0**(2 * n)),
        min(chi_new, chi_0**(4 * n))
    ])
    uC[n] = misc_mera.pad_tensor(uC[n], [
        min(chi_new, chi_0**(2 * n)),
        min(chi_new, chi_0**(2 * n)),
        min(chi_new, chi_0**(2 * n)),
        min(chi_new, chi_0**(2 * n))
    ])

  n = len(wC)
  while not misc_mera.all_same_chi(wC[-1]):
    wC.append(
        misc_mera.pad_tensor(wC[-1], [
            min(chi_new, chi_0**(2 * n)),
            min(chi_new, chi_0**(2 * n)),
            min(chi_new, chi_0**(4 * n))
        ]))
    vC.append(
        misc_mera.pad_tensor(vC[-1], [
            min(chi_new, chi_0**(2 * n)),
            min(chi_new, chi_0**(2 * n)),
            min(chi_new, chi_0**(4 * n))
        ]))
    uC.append(
        misc_mera.pad_tensor(uC[-1], [
            min(chi_new, chi_0**(2 * n)),
            min(chi_new, chi_0**(2 * n)),
            min(chi_new, chi_0**(2 * n)),
            min(chi_new, chi_0**(2 * n))
        ]))
    n += 1

  return wC, vC, uC


def initialize_TFI_hams(dtype=tf.float64):
  """
    initialize a transverse field ising hamiltonian

    Returns:
    ------------------
    (hamBA, hamBA)
    tuple of tf.Tensors
    """
  sX = np.array([[0, 1], [1, 0]])
  sY = np.array([[0, -1j], [1j, 0]])
  sZ = np.array([[1, 0], [0, -1]])

  htemp = -np.kron(sX, sX) - 0.5 * (
      np.kron(sZ, np.eye(2)) + np.kron(np.eye(2), sZ))
  hbig = (0.5 * np.kron(np.eye(4), htemp) +
          np.kron(np.eye(2), np.kron(htemp, np.eye(2))) +
          0.5 * np.kron(htemp, np.eye(4))).reshape(2, 2, 2, 2, 2, 2, 2, 2)

  hamAB = tf.Variable(
      (hbig.transpose(0, 1, 3, 2, 4, 5, 7, 6).reshape(4, 4, 4, 4)).astype(
          dtype.as_numpy_dtype),
      use_resource=True,
      name='hamAB_0',
      dtype=dtype)
  hamBA = tf.Variable(
      (hbig.transpose(1, 0, 2, 3, 5, 4, 6, 7).reshape(4, 4, 4, 4)).astype(
          dtype.as_numpy_dtype),
      use_resource=True,
      name='hamBA_0',
      dtype=dtype)
  return hamAB, hamBA


def initialize_mod_binary_MERA(phys_dim, chi, dtype=tf.float64):
  """
    Parameters:
    -------------------
    phys_dim:         int 
                      Hilbert space dimension of the bottom layer
    chi:              int 
                      maximum bond dimension
    dtype:            tensorflow dtype
                      dtype of the MERA tensors
    Returns:
    -------------------
    (wC, vC, uC, rhoAB, rhoBA)
    wC, vC, uC:      list of tf.Tensor
    rhoAB, rhoBA:    tf.Tensor
    """

  wC, vC, uC = increase_bond_dimension_by_adding_layers(
      chi_new=chi,
      wC=[tf.random_uniform(shape=[phys_dim, phys_dim, phys_dim], dtype=dtype)],
      vC=[tf.random_uniform(shape=[phys_dim, phys_dim, phys_dim], dtype=dtype)],
      uC=[
          tf.random_uniform(
              shape=[phys_dim, phys_dim, phys_dim, phys_dim], dtype=dtype)
      ])
  chi_top = wC[-1].shape[2]
  rhoAB = tf.reshape(
      tf.eye(chi_top * chi_top, dtype=dtype),
      (chi_top, chi_top, chi_top, chi_top))

  rhoBA = tf.reshape(
      tf.eye(chi_top * chi_top, dtype=dtype),
      (chi_top, chi_top, chi_top, chi_top))

  return wC, vC, uC, rhoAB, rhoBA


def run_mod_binary_mera_optimization_TFI(chis=[8, 12, 16],
                                         niters=[200, 300, 1000],
                                         dtype=tf.float64,
                                         verbose=1,
                                         refsym=True):
  wC, vC, uC, rhoAB_0, rhoBA_0 = initialize_mod_binary_MERA(
      phys_dim=4, chi=chis[0], dtype=dtype)
  hamAB_0, hamBA_0 = initialize_TFI_hams(dtype=dtype)
  energies = []
  walltimes = []
  for chi, niter in zip(chis, niters):
    wC, vC, uC = increase_bond_dimension_by_padding(chi, wC, vC, uC)
    rhoAB_0, rhoBA_0 = misc_mera.pad_tensor(
        rhoAB_0, [chi, chi, chi, chi]), misc_mera.pad_tensor(
            rhoBA_0, [chi, chi, chi, chi])
    wC, vC, uC, rhoAB_0, rhoBA_0, times, es = optimize_mod_binary_mera(
        hamAB_0=hamAB_0,
        hamBA_0=hamBA_0,
        rhoAB_0=rhoAB_0,
        rhoBA_0=rhoBA_0,
        wC=wC,
        vC=vC,
        uC=uC,
        verbose=verbose,
        numiter=niter,
        opt_u=True,
        opt_vw=True,
        refsym=refsym)
    energies.extend(es)
    walltimes.extend(times)
  return energies, walltimes, wC, vC, uC


def benchmark_ascending_operator(hab, hba, w, v, u, num_layers):
  t1 = time.time()
  for t in range(num_layers):
    hab, hba = ascending_super_operator(hab, hba, w, v, u, refsym=False)
  return time.time() - t1


def benchmark_descending_operator(rhoab, rhoba, w, v, u, num_layers):
  t1 = time.time()
  for p in range(num_layers):
    rhoab, rhoba = descending_super_operator(
        rhoab, rhoba, w, v, u, refsym=False)
  return time.time() - t1


def run_ascending_operator_benchmark(filename,
                                     chis=[4, 8, 16, 32],
                                     num_layers=8,
                                     dtype=tf.float64,
                                     device=None):
  walltimes = {'warmup': {}, 'profile': {}}
  for chi in chis:
    print('running ascending-operator benchmark for chi = {0} benchmark'.format(
        chi))
    with tf.device(device):
      wC, vC, uC, rhoAB, rhoBA = initialize_mod_binary_MERA(
          phys_dim=4, chi=chi, dtype=dtype)
      shape = uC[-1].shape
      hab, hba = tf.random_uniform(
          shape=shape, dtype=dtype), tf.random_uniform(
              shape=shape, dtype=dtype)
      walltimes['warmup'][chi] = benchmark_ascending_operator(
          hab, hba, wC[-1], vC[-1], uC[-1], num_layers)
      print('     warmup took {0} s'.format(walltimes['warmup'][chi]))
      walltimes['profile'][chi] = benchmark_ascending_operator(
          hab, hba, wC[-1], vC[-1], uC[-1], num_layers)
      print('     profile took {0} s'.format(walltimes['profile'][chi]))

  with open(filename + '.pickle', 'wb') as f:
    pickle.dump(walltimes, f)
  return walltimes


def run_descending_operator_benchmark(filename,
                                      chis=[4, 8, 16, 32],
                                      num_layers=8,
                                      dtype=tf.float64,
                                      device=None):
  walltimes = {'warmup': {}, 'profile': {}}
  for chi in chis:
    print(
        'running descending-operator benchmark for chi = {0} benchmark'.format(
            chi))
    with tf.device(device):
      wC, vC, uC, rhoAB, rhoBA = initialize_mod_binary_MERA(
          phys_dim=4, chi=chi, dtype=dtype)
      shape = uC[-1].shape
      hab, hba = tf.random_uniform(
          shape=shape, dtype=dtype), tf.random_uniform(
              shape=shape, dtype=dtype)
      walltimes['warmup'][chi] = benchmark_descending_operator(
          rhoAB, rhoBA, wC[-1], vC[-1], uC[-1], num_layers=num_layers)
      print('     warmup took {0} s'.format(walltimes['warmup'][chi]))
      walltimes['profile'][chi] = benchmark_descending_operator(
          rhoAB, rhoBA, wC[-1], vC[-1], uC[-1], num_layers=num_layers)
      print('     profile took {0} s'.format(walltimes['profile'][chi]))

  with open(filename + '.pickle', 'wb') as f:
    pickle.dump(walltimes, f)
  return walltimes


def run_optimization_naive_benchmark(filename,
                                     chis=[4, 8, 16, 32],
                                     dtype=tf.float64,
                                     numiter=30,
                                     device=None,
                                     opt_u=True,
                                     opt_vw=True,
                                     np_update=True,
                                     refsym=True):

  walltimes = {'profile': {}, 'energies': {}}
  with tf.device(device):
    for chi in chis:
      print('running naive optimization benchmark for chi = {0}'.format(chi))

      wC, vC, uC, rhoAB_0, rhoBA_0 = initialize_mod_binary_MERA(
          phys_dim=4, chi=chi, dtype=dtype)
      hamAB_0, hamBA_0 = initialize_TFI_hams(dtype=dtype)
      wC, vC, uC, rhoAB_0, rhoBA_0, runtimes, energies = optimize_mod_binary_mera(
          hamAB_0=hamAB_0,
          hamBA_0=hamBA_0,
          rhoAB_0=rhoAB_0,
          rhoBA_0=rhoBA_0,
          wC=wC,
          vC=vC,
          uC=uC,
          verbose=1,
          numiter=numiter,
          opt_u=True,
          opt_vw=True,
          refsym=refsym)
      walltimes['profile'][chi] = runtimes
      walltimes['energies'][chi] = energies
      print('     steps took {0} s'.format(walltimes['profile'][chi]))
      with open(filename + '.pickle', 'wb') as f:
        pickle.dump(walltimes, f)

  return walltimes


def run_optimization_benchmark(filename,
                               chis=[4, 8, 16, 32],
                               numiters=[200, 200, 400, 800],
                               dtype=tf.float64,
                               device=None,
                               refsym=True,
                               verbose=1):

  walltimes = {}
  with tf.device(device):
    print('running optimization benchmark')
    energies, runtimes, wC, vC, uC = run_mod_binary_mera_optimization_TFI(
        chis=chis, niters=numiters, dtype=dtype, verbose=verbose, refsym=refsym)
    walltimes['profile'] = runtimes
    walltimes['energies'] = energies
    with open(filename + '.pickle', 'wb') as f:
      pickle.dump(walltimes, f)

  return walltimes


if __name__ == "__main__":
  if not tf.executing_eagerly():
    pass

  else:
    #uncomment to perform benchmarks
    benchmarks = {
        'descend': {
            'chis': [8, 10, 12],
            'dtype': tf.float32,
            'num_layers': 8
        }
    }
    # benchmarks = {'ascend' : {'chis' :  [8, 10, 12],
    #                            'dtype' : tf.float32,
    #                            'num_layers' : 8}}

    # benchmarks = {'ascend' : {'chis' :  [16, 32, 40, 48, 54],
    #                           'dtype' : tf.float32,
    #                           'num_layers' : 8},
    #               'descend' : {'chis' :  [16, 32, 40, 48, 54],
    #                            'dtype' : tf.float32,
    #                            'num_layers' : 8}}

    # benchmarks = {'optimize_naive' : {'chis' :  [8,12],
    #                                   'dtype' : tf.float64,
    #                                   'opts_u' : [True, True],
    #                                   'opts_vw' : [True, True],
    #                                   'np_update' : True,
    #                                   'refsym' : True,
    #                                   'numiter' : 5}}
    # benchmarks = {'optimize' : {'chis' :  [8, 10, 12, 16],
    #                             'numiters' : [400, 400, 800, 1000],
    #                             'dtype' : tf.float64,
    #                             'refsym' : True}}

    use_gpu = False
    DEVICES = tf.contrib.eager.list_devices()
    print("Available devices:")
    for i, device in enumerate(DEVICES):
      print("%d) %s" % (i, device))
    CPU = '/device:CPU:0'
    GPU = '/job:localhost/replica:0/task:0/device:GPU:0'
    if use_gpu:
      specified_device_type = GPU
      name = 'GPU'
    else:
      specified_device_type = CPU
      name = 'CPU'

    if 'ascend' in benchmarks:
      num_layers = benchmarks['ascend']['num_layers']
      dtype = benchmarks['ascend']['dtype']
      chis = benchmarks['ascend']['chis']
      chis_str = '{0}'.format(chis).replace(' ', '')
      fname = 'ascending_benchmarks'
      if not os.path.exists(fname):
        os.mkdir(fname)
      os.chdir(fname)
      run_ascending_operator_benchmark(
          name +
          'modified_binary_mera_ascending_benchmark_chi{0}_numlayers{1}_dtype{2}'.
          format(chis_str, num_layers, dtype.name),
          chis=chis,
          num_layers=num_layers,
          device=specified_device_type)

    if 'descend' in benchmarks:
      num_layers = benchmarks['descend']['num_layers']
      dtype = benchmarks['descend']['dtype']
      chis = benchmarks['descend']['chis']
      chis_str = '{0}'.format(chis).replace(' ', '')
      fname = 'descending_benchmarks'
      if not os.path.exists(fname):
        os.mkdir(fname)
      os.chdir(fname)
      run_descending_operator_benchmark(
          name +
          'modified_binary_mera_descending_benchmark_chi{0}_numlayers{1}_dtype{2}'.
          format(chis_str, num_layers, dtype.name),
          chis=chis,
          num_layers=num_layers,
          device=specified_device_type)

    if 'optimize_naive' in benchmarks:
      dtype = benchmarks['optimize_naive']['dtype']
      chis = benchmarks['optimize_naive']['chis']
      numiter = benchmarks['optimize_naive']['numiter']
      opts_u = benchmarks['optimize_naive']['opts_u']
      opts_vw = benchmarks['optimize_naive']['opts_vw']
      np_update = benchmarks['optimize_naive']['np_update']
      refsym = benchmarks['optimize_naive']['refsym']
      chis_str = '{0}'.format(chis).replace(' ', '')
      fname = 'benchmarks_optimize_naive'
      if not os.path.exists(fname):
        os.mkdir(fname)
      os.chdir(fname)

      for opt_u, opt_vw in zip(opts_u, opts_vw):
        run_optimization_naive_benchmark(
            name +
            'modified_binary_mera_optimization_benchmark_Nthreads{6}_chi{0}_dtype{1}_opt_u{2}_opt_vw{3}_numiter{4}_npupdate{5}'.
            format(chis_str, dtype.name, opt_u, opt_vw, numiter, np_update,
                   NUM_THREADS),
            chis=chis,
            numiter=numiter,
            device=specified_device_type,
            opt_u=opt_u,
            dtype=dtype,
            opt_vw=opt_vw,
            np_update=np_update,
            refsym=refsym)

    if 'optimize' in benchmarks:
      dtype = benchmarks['optimize']['dtype']
      chis = benchmarks['optimize']['chis']
      numiters = benchmarks['optimize']['numiters']
      refsym = benchmarks['optimize']['refsym']
      chis_str = '{0}'.format(chis).replace(' ', '')
      numiters_str = '{0}'.format(numiters).replace(' ', '')
      fname = 'benchmarks_optimize'
      if not os.path.exists(fname):
        os.mkdir(fname)
      os.chdir(fname)
      filename = name + 'modified_binary_mera_optimization_benchmark_Nthreads{0}_chis{1}_dtype{2}_numiters{3}'.format(
          NUM_THREADS, chis_str, dtype.name, numiters_str)

      run_optimization_benchmark(
          filename,
          chis=chis,
          numiters=numiters,
          dtype=dtype,
          device=specified_device_type,
          refsym=True,
          verbose=1)
