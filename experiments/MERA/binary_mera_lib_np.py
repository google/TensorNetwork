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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import time
import pickle
import lib.ncon as ncon
from sys import stdout

def u_update_svd_numpy(wIn):
    shape = wIn.shape
    ut, st, vt = np.linalg.svd(np.reshape(wIn,(shape[0] * shape[1], shape[2] * shape[3])), full_matrices=False)
    return -np.reshape(ncon.ncon([ut, vt],[[-1,1],[1, -2]]), shape)

def w_update_svd_numpy(wIn):
    shape = wIn.shape
    ut, st, vt = np.linalg.svd(np.reshape(wIn,(shape[0] * shape[1], shape[2])), full_matrices=False)
    return -np.reshape(ncon.ncon([ut, vt],[[-1,1],[1, -2]]), shape)    


def trace(rho):
    dim = len(rho.shape)//2 
    inds = [n + 1 for n in range(dim)]
    return ncon.ncon([rho],[inds+inds])

def symmetrize(rho):
    dim = len(rho.shape) // 2 
    inds_1 = [n  for n in range(dim)]
    inds_2 = [n + dim  for n in range(dim)]
    indices = inds_2 + inds_1
    return 1/2 * (rho + np.conj(np.transpose(rho,indices)))


def pad_tensor(tensor, new_shape):
    paddings = np.zeros((len(tensor.shape),2)).astype(np.int32)
    for n in range(len(new_shape)):
        paddings[n,1] = max(new_shape[n] - tensor.shape[n],0)
    return np.pad(tensor, paddings, mode='constant')

def all_same_chi(*tensors):
    chis = [t.shape[n] for t in tensors for n in range(len(t.shape))]
    return np.all([c == chis[0] for c in chis])

def ascending_super_operator(ham, isometry, unitary):
    return left_ascending_super_operator(ham, isometry, unitary) + right_ascending_super_operator(ham, isometry, unitary)


def left_ascending_super_operator(ham, isometry, unitary):
    inds_right_ul =     [1, 2, 13, 5]
    inds_right_ur =     [6, 9, 7, 16]
    inds_right_ul_c =   [3, 4, 14, 10]
    inds_right_ur_c =   [8, 9, 11, 17]
    
    inds_right_iso_l =  [12, 13, -4]
    inds_right_iso_c =  [5, 7, -5]
    inds_right_iso_r =  [16, 15, -6]
    inds_right_iso_l_c = [12, 14, -1]
    inds_right_iso_c_c = [10, 11, -2]
    inds_right_iso_r_c = [17, 15, -3]
    
    inds_right_ham =    [3, 4, 8, 1, 2, 6]
    
    hright = ncon.ncon([isometry, isometry, isometry, np.conj(isometry), np.conj(isometry), np.conj(isometry),
                       unitary, unitary, np.conj(unitary), np.conj(unitary), ham],
                      [inds_right_iso_l, inds_right_iso_c, inds_right_iso_r,
                       inds_right_iso_l_c, inds_right_iso_c_c, inds_right_iso_r_c,
                       inds_right_ul, inds_right_ur, inds_right_ul_c, inds_right_ur_c,
                       inds_right_ham])
    return hright



def right_ascending_super_operator(ham, isometry, unitary):
    inds_left_ul =     [8, 6, 13, 7]
    inds_left_ur =     [1, 2, 5, 16]
    inds_left_ul_c =   [8, 9, 14, 10]
    inds_left_ur_c =   [3, 4, 11, 17]
    inds_left_iso_l =  [12, 13, -4]
    inds_left_iso_c =  [7, 5, -5]  
    inds_left_iso_r =  [16, 15, -6]
    inds_left_iso_l_c = [12, 14, -1]
    inds_left_iso_c_c = [10, 11, -2]
    inds_left_iso_r_c = [17, 15, -3]
    inds_left_ham =    [9, 3, 4, 6, 1, 2]
    
    hleft = ncon.ncon([isometry, isometry, isometry, np.conj(isometry), np.conj(isometry), np.conj(isometry),
                       unitary, unitary, np.conj(unitary), np.conj(unitary), ham],
                      [inds_left_iso_l, inds_left_iso_c, inds_left_iso_r,
                       inds_left_iso_l_c, inds_left_iso_c_c, inds_left_iso_r_c,
                       inds_left_ul, inds_left_ur, inds_left_ul_c, inds_left_ur_c,
                       inds_left_ham])
    return hleft



def left_descending_super_operator(rho, isometry, unitary):
    inds_left_ul =     [-1,-2, 8, 9]
    inds_left_ur =     [-3, 12, 10, 11]
    inds_left_ul_c =   [-4, -5, 16, 17]
    inds_left_ur_c =   [-6, 12, 14, 13]
    inds_left_iso_l =  [1, 8, 2]
    inds_left_iso_c =  [9, 10, 7]
    inds_left_iso_r =  [11, 4, 5]
    inds_left_iso_l_c = [1, 16, 3]
    inds_left_iso_c_c = [17, 14, 15]
    inds_left_iso_r_c = [13, 4, 6]
    inds_left_rho =    [2, 7, 5, 3, 15, 6]
    rho = ncon.ncon([isometry, isometry, isometry, np.conj(isometry), np.conj(isometry), np.conj(isometry),
                     unitary, unitary, np.conj(unitary), np.conj(unitary), rho],
                    [inds_left_iso_l, inds_left_iso_c, inds_left_iso_r,
                    inds_left_iso_l_c, inds_left_iso_c_c, inds_left_iso_r_c,
                     inds_left_ul, inds_left_ur, inds_left_ul_c, inds_left_ur_c,
                     inds_left_rho])
    return rho


def right_descending_super_operator(rho, isometry, unitary):
    inds_right_ul =     [10, -1, 8, 9]
    inds_right_ur =     [-2, -3, 16, 17]
    inds_right_ul_c =   [10, -4, 11, 13]
    inds_right_ur_c =   [-5, -6, 14, 15]
    
    inds_right_iso_l =  [1, 8, 2]
    inds_right_iso_c =  [9, 16, 7]
    inds_right_iso_r =  [17, 4, 5]
    
    inds_right_iso_l_c = [1, 11, 3]
    inds_right_iso_c_c = [13, 14, 12]
    inds_right_iso_r_c = [15, 4, 6]
    
    inds_right_rho =    [2, 7, 5, 3, 12, 6]
    rho = ncon.ncon([isometry, isometry, isometry,
                     np.conj(isometry), np.conj(isometry), np.conj(isometry),
                     unitary, unitary,
                     np.conj(unitary), np.conj(unitary), rho],
                    [inds_right_iso_l, inds_right_iso_c, inds_right_iso_r,
                     inds_right_iso_l_c, inds_right_iso_c_c, inds_right_iso_r_c,
                     inds_right_ul, inds_right_ur,
                     inds_right_ul_c, inds_right_ur_c,
                     inds_right_rho])
    return rho



def descending_super_operator(rho, isometry, unitary):
    rho_1 = right_descending_super_operator(rho, isometry, unitary)
    rho_2 = left_descending_super_operator(rho, isometry, unitary)    
    rho = 0.5*(rho_1 + rho_2)
    rho = symmetrize(rho)
    rho = rho/ncon.ncon([rho],[[1, 2, 3, 1, 2, 3]])
    return rho


def get_env_disentangler(ham, rho, isometry, unitary):
    inds_1_rho =   [17, 14, 11, 19, 12, 13]
    inds_1_wl =    [16, -3, 17]
    inds_1_wc =    [-4, 15, 14]
    inds_1_wr =    [9, 8, 11]
    inds_1_wl_c =  [16, 18, 19]
    inds_1_wc_c =  [3, 5, 12]
    inds_1_wr_c =  [10, 8, 13]
    inds_1_ur =    [6, 7, 15, 9]
    inds_1_ul_c =  [1, 2, 18, 3]
    inds_1_ur_c =  [4, 7, 5, 10]
    inds_1_ham =   [1, 2, 4, -1, -2, 6]
    
    env_1 = ncon.ncon([rho, isometry, isometry, isometry,
                       np.conj(isometry), np.conj(isometry), np.conj(isometry),
                       unitary, np.conj(unitary), np.conj(unitary),
                       ham],
                      [inds_1_rho, inds_1_wl, inds_1_wc, inds_1_wr,
                       inds_1_wl_c, inds_1_wc_c, inds_1_wr_c,
                       inds_1_ur, inds_1_ul_c, inds_1_ur_c, inds_1_ham])

    inds_2_rho =   [17, 14, 11, 18, 13, 12] 
    inds_2_wl =    [16, -3, 17]
    inds_2_wc =    [-4, 15, 14]
    inds_2_wr =    [6, 5, 11]
    inds_2_wl_c =  [16, 19, 18]
    inds_2_wc_c =  [10, 8, 13]
    inds_2_wr_c =  [7, 5, 12]
    
    inds_2_ur =    [1, 2, 15, 6]
    inds_2_ul_c =  [-1, 9, 19, 10]
    inds_2_ur_c =  [3, 4, 8, 7]
    inds_2_ham =   [9, 3, 4, -2, 1, 2]
    
    env_2 = ncon.ncon([rho, isometry, isometry, isometry,
                       np.conj(isometry), np.conj(isometry), np.conj(isometry),
                       unitary, np.conj(unitary), np.conj(unitary),
                       ham],
                      [inds_2_rho, inds_2_wl, inds_2_wc, inds_2_wr,
                       inds_2_wl_c, inds_2_wc_c, inds_2_wr_c,
                       inds_2_ur, inds_2_ul_c, inds_2_ur_c, inds_2_ham])



    inds_3_rho =   [11, 18, 17, 12, 13, 14]
    inds_3_wl =    [5, 6, 11]
    inds_3_wc =    [19, -3, 18]
    inds_3_wr =    [-4, 16, 17]
    inds_3_wl_c =  [5, 7, 12]
    inds_3_wc_c =  [8, 10, 13]
    inds_3_wr_c =  [15, 16, 14]
    inds_3_ul =    [1, 2, 6, 19]
    inds_3_ul_c =  [3, 4, 7, 8]
    inds_3_ur_c =  [9, -2, 10, 15]
    inds_3_ham =   [3, 4, 9, 1, 2, -1]
    
    env_3 = ncon.ncon([rho, isometry, isometry, isometry,
                      np.conj(isometry), np.conj(isometry), np.conj(isometry),
                      unitary, np.conj(unitary), np.conj(unitary),
                      ham],
                     [inds_3_rho, inds_3_wl, inds_3_wc, inds_3_wr,
                      inds_3_wl_c, inds_3_wc_c, inds_3_wr_c,
                      inds_3_ul, inds_3_ul_c, inds_3_ur_c, inds_3_ham])


    inds_4_rho =   [11, 14, 17, 12, 13, 18]
    inds_4_wl =    [8, 9, 11]
    inds_4_wc =    [15, -3, 14]
    inds_4_wr =    [-4, 16, 17]
    inds_4_wl_c =  [8, 10, 12]
    inds_4_wc_c =  [5, 3, 13]
    inds_4_wr_c =  [19, 16, 18]
    inds_4_ul =    [6, 7,  9, 15]
    inds_4_ul_c =  [6, 4, 10, 5]
    inds_4_ur_c =  [1, 2, 3, 19]
    inds_4_ham =   [4, 1, 2, 7, -1, -2]
    
    env_4 = ncon.ncon([rho, isometry, isometry, isometry,
                      np.conj(isometry), np.conj(isometry), np.conj(isometry),
                      unitary, np.conj(unitary), np.conj(unitary),
                      ham],
                     [inds_4_rho, inds_4_wl, inds_4_wc, inds_4_wr,
                      inds_4_wl_c, inds_4_wc_c, inds_4_wr_c,
                      inds_4_ul, inds_4_ul_c, inds_4_ur_c, inds_4_ham])
    
    return env_1 + env_2 + env_3 + env_4




def get_env_isometry(ham, rho, isometry, unitary):
    inds_1_wc = [5, 6, 15]
    inds_1_wr = [13, 12, 16]
    
    inds_1_wl_c = [-1, 20, 19]
    inds_1_wc_c = [10, 11, 17]
    inds_1_wr_c = [14, 12, 18]
    
    inds_1_ul = [1, 2, -2, 5]
    inds_1_ur = [7, 9, 6, 13]
    
    inds_1_ul_c = [3, 4, 20, 10]    
    inds_1_ur_c = [8, 9, 11, 14]
    
    inds_1_ham = [3, 4, 8, 1, 2, 7]
    inds_1_rho = [-3, 15, 16, 19, 17, 18]    


    env_1 = ncon.ncon([isometry, isometry,
                       np.conj(isometry), np.conj(isometry), np.conj(isometry),
                       unitary, unitary, np.conj(unitary), np.conj(unitary),
                       ham, rho],
                      [inds_1_wc, inds_1_wr,
                       inds_1_wl_c, inds_1_wc_c, inds_1_wr_c,
                       inds_1_ul, inds_1_ur, inds_1_ul_c, inds_1_ur_c,
                       inds_1_ham, inds_1_rho])
    
    #########################

    inds_2_wc = [14, 8, 9]
    inds_2_wr = [6, 5, 10]
    
    inds_2_wl_c = [-1, 19, 20]
    inds_2_wc_c = [18, 12, 13]
    inds_2_wr_c = [7, 5, 11]
    
    inds_2_ul = [16, 15, -2, 14]
    inds_2_ur = [1, 2, 8, 6]
    
    inds_2_ul_c = [16, 17, 19, 18]
    inds_2_ur_c = [3, 4, 12, 7]
    
    inds_2_ham = [17, 3, 4, 15, 1, 2]
    inds_2_rho = [-3, 9, 10, 20, 13, 11]    

    env_2 = ncon.ncon([isometry, isometry,
                       np.conj(isometry), np.conj(isometry), np.conj(isometry),
                       unitary, unitary, np.conj(unitary), np.conj(unitary),
                       ham, rho],
                      [inds_2_wc, inds_2_wr,
                       inds_2_wl_c, inds_2_wc_c, inds_2_wr_c,
                       inds_2_ul, inds_2_ur, inds_2_ul_c, inds_2_ur_c,
                       inds_2_ham, inds_2_rho])

    #########################
    inds_3_wl = [5, 6, 9]
    inds_3_wr = [20, 17, 16]
    
    inds_3_wl_c = [5, 7, 10]
    inds_3_wc_c = [8, 13, 11]
    inds_3_wr_c = [14, 17, 15]
    
    inds_3_ul = [1, 2, 6, -1]
    inds_3_ur = [18, 19, -2, 20]
    
    inds_3_ul_c = [3, 4, 7, 8]
    inds_3_ur_c = [12, 19, 13, 14]
    
    inds_3_ham = [3, 4, 12, 1, 2, 18]
    inds_3_rho = [9, -3, 16, 10, 11, 15]    
    env_3 = ncon.ncon([isometry, isometry,
                       np.conj(isometry), np.conj(isometry), np.conj(isometry),
                       unitary, unitary, np.conj(unitary), np.conj(unitary),
                       ham, rho],
                      [inds_3_wl, inds_3_wr,
                       inds_3_wl_c, inds_3_wc_c, inds_3_wr_c,
                       inds_3_ul, inds_3_ur, inds_3_ul_c, inds_3_ur_c,
                       inds_3_ham, inds_3_rho])


    #########################
    inds_4_wl = [16, 17, 19]
    inds_4_wr = [6, 5, 9]
    
    inds_4_wl_c = [16, 18, 20]
    inds_4_wc_c = [13, 8, 10]
    inds_4_wr_c = [7, 5, 11]
    
    inds_4_ul = [14, 15, 17, -1]
    inds_4_ur = [1, 2, -2, 6]
    
    inds_4_ul_c = [14, 12, 18, 13]    
    inds_4_ur_c = [3, 4, 8, 7]
    
    inds_4_ham = [12, 3, 4, 15, 1, 2]
    inds_4_rho = [19, -3, 9, 20, 10, 11]    

    env_4 = ncon.ncon([isometry, isometry,
                       np.conj(isometry), np.conj(isometry), np.conj(isometry),
                       unitary, unitary, np.conj(unitary), np.conj(unitary),
                       ham, rho],
                      [inds_4_wl, inds_4_wr,
                       inds_4_wl_c, inds_4_wc_c, inds_4_wr_c,
                       inds_4_ul, inds_4_ur, inds_4_ul_c, inds_4_ur_c,
                       inds_4_ham, inds_4_rho])


    #########################
    inds_5_wl = [5, 6, 9]
    inds_5_wc = [18, 19, 20]
    
    inds_5_wl_c = [5, 7, 10]
    inds_5_wc_c = [8, 13, 11]
    inds_5_wr_c = [14, -2, 15]
    
    inds_5_ul = [1, 2, 6, 18]
    inds_5_ur = [16, 17, 19, -1]
    
    inds_5_ul_c = [3, 4, 7, 8]    
    inds_5_ur_c = [12, 17, 13, 14]
    
    inds_5_ham = [3, 4, 12, 1, 2, 16]
    inds_5_rho = [9, 20, -3, 10, 11, 15]    

    env_5 = ncon.ncon([isometry, isometry,
                       np.conj(isometry), np.conj(isometry), np.conj(isometry),
                       unitary, unitary, np.conj(unitary), np.conj(unitary),
                       ham, rho],
                      [inds_5_wl, inds_5_wc,
                       inds_5_wl_c, inds_5_wc_c, inds_5_wr_c,
                       inds_5_ul, inds_5_ur, inds_5_ul_c, inds_5_ur_c,
                       inds_5_ham, inds_5_rho])


    #########################
    inds_6_wl = [12, 13, 15]
    inds_6_wc = [6, 5, 16]
    
    inds_6_wl_c = [12, 14, 17]
    inds_6_wc_c = [10, 11, 18]
    inds_6_wr_c = [20, -2, 19]
    
    inds_6_ul = [8, 7, 13, 6]
    inds_6_ur = [1, 2, 5, -1]
    
    inds_6_ul_c = [8, 9, 14, 10]    
    inds_6_ur_c = [3, 4, 11, 20]
    
    inds_6_ham = [9, 3, 4, 7, 1, 2]
    inds_6_rho = [15, 16, -3, 17, 18, 19]    

    env_6 = ncon.ncon([isometry, isometry,
                       np.conj(isometry), np.conj(isometry), np.conj(isometry),
                       unitary, unitary, np.conj(unitary), np.conj(unitary),
                       ham, rho],
                      [inds_6_wl, inds_6_wc,
                       inds_6_wl_c, inds_6_wc_c, inds_6_wr_c,
                       inds_6_ul, inds_6_ur, inds_6_ul_c, inds_6_ur_c,
                       inds_6_ham, inds_6_rho])


    return env_1 + env_2 + env_3 + env_4 + env_5 + env_6



def steady_state_density_matrices(nsteps, rho, isometry, unitary, verbose=0):
    """
    obtain steady state density matrix of the scale invariant binary MERA
    Parameters:
    """
    for n in range(nsteps):
        if verbose > 0:
            stdout.write('\r binary-mera stead-state rdm iteration = %i' %(n))
            stdout.flush()
        rho_new = descending_super_operator(rho, isometry, unitary)
        rho_new = symmetrize(rho_new)
        rho_new = rho_new/ncon.ncon([rho_new],[[1, 2, 3, 1, 2, 3]])
        rho = rho_new
    return rho

def unlock_layer(wC, uC, noise=0.0):
    wC.append(copy.copy(wC[-1]))
    uC.append(copy.copy(uC[-1]))
    wC[-1] += (np.random.random_sample(shape=wC[-1].shape).astype(wC[-1].dtype) - 0.5) * noise
    uC[-1] += (np.random.random_sample(shape=uC[-1].shape).astype(uC[-1].dtype) - 0.5) * noise
    return wC, uC

def increase_bond_dimension_by_adding_layers(chi_new, wC, uC):
    """
    increase the bond dimension of the MERA to `chi_new`
    by padding tensors in the last layer with zeros. If the desired `chi_new` cannot
    be obtained from padding, adds layers of Tensors
    the last layer is guaranteed to have uniform bond dimension
    Parameters:
    --------------------------------
    chi_new:         int 
                     new bond dimenion
    wC, uC:         list of np.ndarray 
                     MERA isometries and disentanglers
    Returns: 
    --------------------------------
    (wC, uC):       list of np.ndarrays
    
    """
    if all_same_chi(wC[-1], uC[-1])  and (wC[-1].shape[2] >= chi_new):
        #nothing to do here
        return wC, uC
    elif all_same_chi(wC[-1], uC[-1])  and (wC[-1].shape[2] < chi_new):    
        chi = min(chi_new, wC[-1].shape[0] * wC[-1].shape[1])
        wC[-1] = pad_tensor(wC[-1], [wC[-1].shape[0], wC[-1].shape[1], chi])
        wC_temp = copy.deepcopy(wC[-1])
        uC_temp = copy.deepcopy(uC[-1])
        wC.append(pad_tensor(wC_temp, [chi, chi, chi]))
        uC.append(pad_tensor(uC_temp, [chi, chi, chi, chi]))
        return increase_bond_dimension_by_adding_layers(chi_new, wC, uC)            

    elif not all_same_chi(wC[-1], uC[-1]):
        raise ValueError('chis of last layer have to be all the same!')


def pad_mera_tensors(chi_new, wC, uC):
    """
    increase the bond dimension of the MERA to `chi_new`
    by padding tensors in all layers with zeros. If the desired `chi_new` cannot
    be obtained from padding, adds layers of Tensors
    the last layer is guaranteed to have uniform bond dimension
    Parameters:
    --------------------------------
    chi_new:         int 
                     new bond dimenion
    wC, uC:          list of np.ndarray 
                     MERA isometries and disentanglers
    Returns: 
    --------------------------------
    (wC, uC):       list of np.ndarrays
    """

    all_chis = [t.shape[n] for t in wC for n in range(len(t.shape))]
    if not np.all([c <= chi_new for c in all_chis]):
        #nothing to increase
        return wC, uC
    
    chi_0 = wC[0].shape[0]
    wC[0] = pad_tensor(wC[0], [chi_0, chi_0, min(chi_new, chi_0 ** 2)])
    
    for n in range(1, len(wC)):
        wC[n] = pad_tensor(wC[n], [min(chi_new,chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** (n + 1)))])
        uC[n] = pad_tensor(uC[n], [min(chi_new,chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)), min(chi_new,chi_0 ** (2 ** n)), min(chi_new,chi_0 ** (2 ** n))])

    n = len(wC)
    while not all_same_chi(wC[-1]):
        wC.append(pad_tensor(wC[-1], [min(chi_new,chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** (n + 1)))]))
        uC.append(pad_tensor(uC[-1], [min(chi_new,chi_0 ** (2 ** n)), min(chi_new, chi_0 ** (2 ** n)), min(chi_new,chi_0 ** (2 ** n)), min(chi_new,chi_0 ** (2 ** n))]))  
        n +=1

    return wC, uC

def initialize_binary_MERA(phys_dim,
                           chi,
                           dtype=np.float64):
    """
    initialize a binary MERA network of bond dimension `chi`
    
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
    (wC, uC, rho)
    wC, uC:      list of np.ndarray
    rho:         np.ndarray
    """
    wC = []
    uC = []    
    n = 0
    while True:
        wC.append(np.reshape(np.eye(phys_dim ** (2 ** (n + 1)), min(phys_dim ** (2 ** (n + 1)), chi)).astype(dtype),
                             (phys_dim ** (2 ** n), phys_dim ** (2 ** n), min(phys_dim ** (2 ** (n + 1)), chi))))
        uC.append(np.reshape(np.eye(phys_dim ** (2 ** (n + 1))).astype(dtype),
                             (phys_dim ** (2 ** n), phys_dim ** (2 ** n), phys_dim ** (2 ** n), phys_dim ** (2 ** n))))
        n += 1
        if all_same_chi(wC[-1]):
            break


    chi_top = wC[-1].shape[2]
    rho = np.reshape(np.eye(chi_top * chi_top* chi_top).astype(dtype=dtype),
                     (chi_top, chi_top, chi_top, chi_top, chi_top, chi_top))

    
    return wC, uC, rho/trace(rho)


def initialize_TFI_hams(dtype=np.float64):
    """
    initialize a transverse field ising hamiltonian

    Returns:
    ------------------
    (hamBA, hamBA)
    tuple of np.ndarrays
    """

    sX = np.array([[0, 1], [1, 0]]).astype(dtype)
    sZ = np.array([[1, 0], [0, -1]]).astype(dtype)
    eye = np.eye(2).astype(dtype)
    ham = ncon.ncon([sX, sX, eye],[[-4, -1], [-5, -2], [-6, -3]])+\
        ncon.ncon([sZ, eye, eye],[[-4, -1], [-5, -2], [-6, -3]])/2+\
        ncon.ncon([eye, sZ, eye],[[-4, -1], [-5, -2], [-6, -3]])/2
    return ham


