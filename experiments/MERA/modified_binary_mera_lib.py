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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../../')
import tensorflow as tf
import copy
import numpy as np
import time
import pickle
import ncon as ncon
import misc_mera
from sys import stdout

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
    
    indList1 = [[6, 4, 1, 2], [1, 3, -3], [6, 7, -1], [2, 5, 3, 9],
                [4, 5, 7, 10], [8, 9, -4], [8, 10, -2]]
    indList2 = [[3, 4, 1, 2], [5, 6, -3], [5, 7, -1], [1, 2, 6, 9],
                [3, 4, 7, 10], [8, 9, -4], [8, 10, -2]]
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
    hamABout = ncon.ncon([
        hamBA, v_isometry,
        tf.conj(v_isometry), w_isometry,
        tf.conj(w_isometry)
    ], indList4)

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
    indList3 = [[3, 9, 2, 4], [1, 5, 2], [1, 8, 3], [7, -3, 5, 6],
                [7, -1, 8, 10], [-4, 6, 4], [-2, 10, 9]]
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

    rhoBAout = rhoBAout + 0.5 * ncon.ncon([
        rhoAB, v_isometry,
        tf.conj(v_isometry), w_isometry,
        tf.conj(w_isometry)
    ], indList4)

    return rhoABout, rhoBAout


@tf.contrib.eager.defun
def get_env_disentangler(hamAB,hamBA,rhoBA,w,v,u,refsym):

    indList1 = [[7,8,10,-1],[4,3,9,2],[10,-3,9],[7,5,4],[8,-2,5,6],[1,-4,2],[1,6,3]]
    indList2 = [[7,8,-1,-2],[3,6,2,5],[1,-3,2],[1,9,3],[7,8,9,10],[4,-4,5],[4,10,6]]
    indList3 = [[7,8,-2,10],[3,4,2,9],[1,-3,2],[1,5,3],[-1,7,5,6],[10,-4,9],[8,6,4]]

    uEnv = ncon.ncon([hamAB,rhoBA,w,tf.conj(w),tf.conj(u),v,tf.conj(v)],indList1)
    if refsym:
        uEnv = uEnv + tf.transpose(uEnv,(1,0,3,2))
    else:
        uEnv = uEnv + ncon.ncon([hamAB,rhoBA,w,tf.conj(w),tf.conj(u),v,tf.conj(v)],indList3)
    
    uEnv = uEnv + ncon.ncon([hamBA,rhoBA,w,tf.conj(w),tf.conj(u),v,tf.conj(v)],indList2)

    return uEnv

@tf.contrib.eager.defun
def get_env_w_isometry(hamAB, hamBA, rhoBA, rhoAB, w_isometry, v_isometry, unitary):
    """
    Parameters:
    """
    indList1 = [[7,8,-1,9],[4,3,-3,2],[7,5,4],[9,10,-2,11],[8,10,5,6],[1,11,2],[1,6,3]]
    indList2 = [[1,2,3,4],[10,7,-3,6],[-1,11,10],[3,4,-2,8],[1,2,11,9],[5,8,6],[5,9,7]]
    indList3 = [[5,7,3,1],[10,9,-3,8],[-1,11,10],[4,3,-2,2],[4,5,11,6],[1,2,8],[7,6,9]]
    indList4 = [[3,7,2,-1],[5,6,4,-3],[2,1,4],[3,1,5],[7,-2,6]]

    wEnv = ncon.ncon([hamAB,rhoBA,tf.conj(w_isometry),unitary,tf.conj(unitary),v_isometry,tf.conj(v_isometry)],
                indList1)
    wEnv = wEnv + ncon.ncon([hamBA,rhoBA,tf.conj(w_isometry),unitary,tf.conj(unitary),v_isometry,tf.conj(v_isometry)],
                       indList2)
    
    wEnv = wEnv + ncon.ncon([hamAB,rhoBA,tf.conj(w_isometry),unitary,tf.conj(unitary),v_isometry,tf.conj(v_isometry)],
                       indList3)
    
    wEnv = wEnv + ncon.ncon([hamBA,rhoAB,v_isometry,tf.conj(v_isometry),tf.conj(w_isometry)],
                       indList4)

    return wEnv

@tf.contrib.eager.defun
def get_env_v_isometry(hamAB, hamBA, rhoBA, rhoAB, w_isometry, v_isometry, unitary):

    indList1 = [[6,4,1,3],[9,11,8,-3],[1,2,8],[6,7,9],[3,5,2,-2],[4,5,7,10],[-1,10,11]]
    indList2 = [[3,4,1,2],[8,10,9,-3],[5,6,9],[5,7,8],[1,2,6,-2],[3,4,7,11],[-1,11,10]]
    indList3 = [[9,10,11,-1],[3,4,2,-3],[1,8,2],[1,5,3],[7,11,8,-2],[7,9,5,6],[10,6,4]]
    indList4 = [[7,5,-1,4],[6,3,-3,2],[7,-2,6],[4,1,2],[5,1,3]]

    vEnv = ncon.ncon([hamAB,rhoBA,w_isometry,tf.conj(w_isometry),unitary,tf.conj(unitary),tf.conj(v_isometry)],indList1)
    vEnv = vEnv + ncon.ncon([hamBA,rhoBA,w_isometry,tf.conj(w_isometry),unitary,tf.conj(unitary),tf.conj(v_isometry)],indList2)
    vEnv = vEnv + ncon.ncon([hamAB,rhoBA,w_isometry,tf.conj(w_isometry),unitary,tf.conj(unitary),tf.conj(v_isometry)],indList3)
    vEnv = vEnv + ncon.ncon([hamBA,rhoAB,tf.conj(v_isometry),w_isometry,tf.conj(w_isometry)],indList4)

    return vEnv


@tf.contrib.eager.defun
def steady_state_density_matrices(nsteps, rhoAB, rhoBA, w_isometry, v_isometry, unitary, refsym):
    for n in range(nsteps):
        rhoAB, rhoBA = descending_super_operator(rhoAB, rhoBA, w_isometry, v_isometry, unitary,
                                                 refsym)
        rhoAB = 1/2 * (rhoAB + tf.conj(tf.transpose(rhoAB,(2,3,0,1))))/ncon.ncon([rhoAB],[[1,2,1,2]])
        rhoBA = 1/2 * (rhoBA + tf.conj(tf.transpose(rhoBA,(2,3,0,1))))/ncon.ncon([rhoBA],[[1,2,1,2]])
        if refsym:
            rhoAB = 0.5 * rhoAB + 0.5 * tf.transpose(rhoAB,(1,0,3,2))
            rhoBA = 0.5 * rhoBA + 0.5 * tf.transpose(rhoBA, (1,0,3,2))
    return rhoAB, rhoBA




def optimize_mod_binary_mera(hamAB_0,
                             hamBA_0,
                             rhoAB_0,
                             rhoBA_0,
                             wC, vC, uC,
                             numiter=1000,
                             refsym=True,
                             nsteps_steady_state=8,
                             verbose=0,
                             opt_u=True,
                             opt_vw=True,
                             numpy_update=True,
                             opt_all_layers=False,
                             opt_u_after=9):
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
    numpy_update:          bool
                           if True, use numpy svd to calculate update of disentanglers
    opt_all_layers:        bool
                           if True, optimize all layers
                           if False, optimize only truncating layers
    opt_u_after:           int 
                           start optimizing disentangler only after `opt_u_after` initial optimization steps
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
    
    bias = tf.math.reduce_max(tf.linalg.eigvalsh(tf.reshape(hamAB[0],(chi1 * chi1, chi1 * chi1))))
    hamAB[0] = hamAB[0] - bias * tf.reshape(tf.eye(chi1 * chi1, dtype=dtype), (chi1, chi1, chi1, chi1))
    hamBA[0] = hamBA[0] - bias * tf.reshape(tf.eye(chi1 * chi1, dtype=dtype), (chi1, chi1, chi1, chi1))
    
    skip_layer = [misc_mera.skip_layer(w) for w in wC]
    for p in range(len(wC)):
        if skip_layer[p]:
            hamAB[p+1], hamBA[p+1] = ascending_super_operator(hamAB[p],hamBA[p],wC[p],vC[p],uC[p],refsym)            

    Energies = []
    run_times = []
    for k in range(numiter):
        t1 = time.time()
        rhoAB_0, rhoBA_0 = steady_state_density_matrices(nsteps_steady_state, rhoAB_0, rhoBA_0, wC[-1], vC[-1], uC[-1], refsym)
        rhoAB[-1] = rhoAB_0
        rhoBA[-1] = rhoBA_0        
        for p in range(len(rhoAB)-2,-1,-1):
            rhoAB[p], rhoBA[p] = descending_super_operator(rhoAB[p+1],rhoBA[p+1],wC[p],vC[p],uC[p],refsym)

        if verbose > 0:
            if np.mod(k,10) == 1:
                Energies.append((ncon.ncon([rhoAB[0],hamAB[0]],[[1,2,3,4],[1,2,3,4]]) + 
                                ncon.ncon([rhoBA[0],hamBA[0]],[[1,2,3,4],[1,2,3,4]]))/4 + bias/2)
                stdout.write('\rIteration: %i of %i: E = %.8f, err = %.16f at D = %i with %i layers' %
                             (int(k),int(numiter), float(Energies[-1]), float(Energies[-1] + 4/np.pi,), int(wC[-1].shape[2]), len(wC)))
                stdout.flush()
                
        for p in range(len(wC)):
            if (not opt_all_layers) and  skip_layer[p]:
                continue
            
            if k >= opt_u_after:
                uEnv = get_env_disentangler(hamAB[p],hamBA[p],rhoBA[p+1],wC[p],vC[p],uC[p],refsym)
                if opt_u:
                    if refsym:
                        uEnv = uEnv + tf.transpose(uEnv,(1,0,3,2))
                    if numpy_update:
                        uC[p] = misc_mera.u_update_svd_numpy(uEnv)
                    else:
                        uC[p] = misc_mera.u_update_svd(uEnv)
            
            wEnv = get_env_w_isometry(hamAB[p],hamBA[p],rhoBA[p+1],rhoAB[p+1],wC[p],vC[p],uC[p])
            if opt_vw:            
                if numpy_update:
                    wC[p] = misc_mera.w_update_svd_numpy(wEnv)                
                else:
                    wC[p] = misc_mera.w_update_svd(wEnv)
                if refsym:
                    vC[p] = wC[p]
                else:
                    vEnv = get_env_v_isometry(hamAB[p],hamBA[p],rhoBA[p+1],rhoAB[p+1],wC[p],vC[p],uC[p])
                    vC[p] = misc_mera.w_update_svd(vEnv)
                    
            hamAB[p+1], hamBA[p+1] = ascending_super_operator(hamAB[p],hamBA[p],wC[p],vC[p],uC[p],refsym)
            
        run_times.append(time.time() - t1)
        if verbose > 2:
            print('time per iteration: ',run_times[-1])
            
    return wC, vC, uC, rhoAB[-1], rhoBA[-1], run_times, Energies


def unlock_layer(wC, vC, uC, noise=0.0):
    """
    unlock a layer of the MERA
    essentially it just copies the last layer of the MERA and adds it as a new layer
    Parameters:
    ---------------------
    wC, vC, uC:   list of tf.Tensor 
                  the MERA tensors 
    noise:        float  
                  noise amplitude in the added layer
    Returns:
    (wC, vC, uC): each a list of tf.Tensor 
                  the new MERA tensors
    """
    wC.append(copy.copy(wC[-1]))
    vC.append(copy.copy(wC[-1]))    
    uC.append(copy.copy(uC[-1]))    
    wC[-1] += (tf.random_uniform(shape=wC[-1].shape, minval=-1, maxval=1, dtype=wC[-1].dtype) * noise)
    vC[-1] += (tf.random_uniform(shape=wC[-1].shape, minval=-1, maxval=1, dtype=wC[-1].dtype) * noise)    
    uC[-1] += (tf.random_uniform(shape=uC[-1].shape, minval=-1, maxval=1, dtype=uC[-1].dtype) * noise)
    return wC, vC, uC


def increase_bond_dimension_by_adding_layers(chi_new, wC, vC, uC):
    """
    deprecated
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
    if misc_mera.all_same_chi(wC[-1], vC[-1], uC[-1])  and (wC[-1].shape[2] >= chi_new):
        #nothing to do here
        return wC, vC, uC
    elif misc_mera.all_same_chi(wC[-1], vC[-1], uC[-1])  and (wC[-1].shape[2] < chi_new):    
        chi = min(chi_new, wC[-1].shape[0] * wC[-1].shape[1])
        wC[-1] = misc_mera.pad_tensor(wC[-1], [wC[-1].shape[0], wC[-1].shape[1], chi])
        vC[-1] = misc_mera.pad_tensor(vC[-1], [vC[-1].shape[0], vC[-1].shape[1], chi])
        wC_temp = copy.deepcopy(wC[-1])
        vC_temp = copy.deepcopy(vC[-1])
        uC_temp = copy.deepcopy(uC[-1])
        wC.append(misc_mera.pad_tensor(wC_temp, [chi, chi, chi]))
        vC.append(misc_mera.pad_tensor(vC_temp, [chi, chi, chi]))
        uC.append(misc_mera.pad_tensor(uC_temp, [chi, chi, chi, chi]))
        return increase_bond_dimension_by_adding_layers(chi_new, wC, vC, uC)            

    elif not misc_mera.all_same_chi(wC[-1], vC[-1], uC[-1]):
        raise ValueError('chis of last layer have to be all the same!')


def pad_mera_tensors(chi_new, wC, vC, uC, noise=0.0):
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
    wC[0] = misc_mera.pad_tensor(wC[0], [chi_0, chi_0, min(chi_new, chi_0 ** 2)])
    vC[0] = misc_mera.pad_tensor(vC[0], [chi_0, chi_0, min(chi_new, chi_0 ** 2)])
    
    for n in range(1, len(wC)):
        wC[n] = misc_mera.pad_tensor(wC[n], [min(chi_new,chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** (n + 1)))])
        vC[n] = misc_mera.pad_tensor(vC[n], [min(chi_new,chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** (n + 1)))])
        uC[n] = misc_mera.pad_tensor(uC[n], [min(chi_new,chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)), min(chi_new,chi_0 ** (2 ** n)), min(chi_new,chi_0 ** (2 ** n))])
        wC[n] += (tf.random_uniform(shape=wC[n].shape, dtype=wC[n].dtype) * noise)
        vC[n] += (tf.random_uniform(shape=vC[n].shape, dtype=vC[n].dtype) * noise)        
        uC[n] += (tf.random_uniform(shape=uC[n].shape, dtype=uC[n].dtype) * noise)

    n = len(wC)
    while not misc_mera.all_same_chi(wC[-1]):
        wC.append(misc_mera.pad_tensor(wC[-1], [min(chi_new,chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** (n + 1)))]))
        vC.append(misc_mera.pad_tensor(vC[-1], [min(chi_new,chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** (n + 1)))]))
        uC.append(misc_mera.pad_tensor(uC[-1], [min(chi_new,chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)),
                                                min(chi_new, chi_0 ** (2 ** n))]))
        wC[-1] += (tf.random_uniform(shape=wC[-1].shape, minval=-1, maxval=1, dtype=wC[-1].dtype) * noise)
        vC[-1] += (tf.random_uniform(shape=vC[-1].shape, minval=-1, maxval=1, dtype=vC[-1].dtype) * noise)        
        uC[-1] += (tf.random_uniform(shape=uC[-1].shape, minval=-1, maxval=1, dtype=uC[-1].dtype) * noise)
        
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

    htemp = -np.kron(
        sX, sX) - 0.5 * (np.kron(sZ, np.eye(2)) + np.kron(np.eye(2), sZ))
    hbig = (0.5 * np.kron(np.eye(4), htemp) + np.kron(
        np.eye(2), np.kron(htemp, np.eye(2))) +
            0.5 * np.kron(htemp, np.eye(4))).reshape(2, 2, 2, 2, 2, 2, 2, 2)

    hamAB = tf.Variable(
        (hbig.transpose(0, 1, 3, 2, 4, 5, 7,
                        6).reshape(4, 4, 4, 4)).astype(dtype.as_numpy_dtype),
        use_resource=True,
        name='hamAB_0',
        dtype=dtype)
    hamBA = tf.Variable(
        (hbig.transpose(1, 0, 2, 3, 5, 4, 6,
                        7).reshape(4, 4, 4, 4)).astype(dtype.as_numpy_dtype),
        use_resource=True,
        name='hamBA_0',
        dtype=dtype)
    return hamAB, hamBA


def initialize_mod_binary_MERA(phys_dim,
                               chi,
                               dtype=tf.float64):
                          
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
    wC = []
    vC = []    
    uC = []    
    n = 0
    while True:
        wC.append(tf.reshape(tf.eye(min(phys_dim ** (2 ** (n + 1)), chi ** 2), min(phys_dim ** (2 ** (n + 1)), chi), dtype=dtype),
                             (min(phys_dim ** (2 ** n), chi), min(phys_dim ** (2 ** n), chi), min(phys_dim ** (2 ** (n + 1)), chi))))
        vC.append(tf.reshape(tf.eye(min(phys_dim ** (2 ** (n + 1)), chi ** 2), min(phys_dim ** (2 ** (n + 1)), chi), dtype=dtype),
                             (min(phys_dim ** (2 ** n), chi), min(phys_dim ** (2 ** n), chi), min(phys_dim ** (2 ** (n + 1)), chi))))
        uC.append(tf.reshape(tf.eye(min(phys_dim ** (2 ** (n + 1)), chi **  2), dtype=dtype),
                             (min(phys_dim ** (2 ** n), chi), min(phys_dim ** (2 ** n), chi), min(phys_dim ** (2 ** n), chi), min(phys_dim ** (2 ** n), chi))))
        n += 1
        if misc_mera.all_same_chi(wC[-1]):
            break

    chi_top = wC[-1].shape[2]
    rhoAB = tf.reshape(tf.eye(chi_top * chi_top, dtype=dtype),
                       (chi_top, chi_top, chi_top, chi_top))

    rhoBA = tf.reshape(tf.eye(chi_top * chi_top, dtype=dtype),
                       (chi_top, chi_top, chi_top, chi_top))
    
    return wC, vC, uC, rhoAB, rhoBA
