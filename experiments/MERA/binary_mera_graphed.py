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

NUM_THREADS = 1
import os
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

import tensorflow as tf
import copy
import datetime
import numpy as np
import ncon
import time
import pickle
import tensornetwork as tn
import experiments.MERA.binary_mera_lib as bml
import experiments.MERA.misc_mera as misc_mera
from sys import stdout

config = tf.ConfigProto()
config.intra_op_parallelism_threads = NUM_THREADS
config.inter_op_parallelism_threads = 1

def u_update_svd(disentangler):
    """
    obtain the update to the disentangler using numpy svd
    """
    shape = disentangler.shape
    ut, st, vt = np.linalg.svd(
        np.reshape(disentangler, (shape[0] * shape[1], shape[2] * shape[3])),
        full_matrices=False)
    return -np.reshape(ut.dot(vt), shape)
def w_update_svd(isometry):
    """
    obtain the update to the isometry using numpy svd
    """
    shape = isometry.shape
    ut, st, vt = np.linalg.svd(
        np.reshape(isometry, (shape[0] * shape[1], shape[2])), full_matrices=False)
    return -np.reshape(ut.dot(vt),shape)



def optimize_binary_mera_graphed_TFI(device,
                                     chi=4,
                                     dtype=tf.float64,
                                     numiter=1000,
                                     nsteps_steady_state=10,
                                     verbose=0,
                                     opt_u_after=30,
                                     E_exact=-4 / np.pi):
    """
    optimization of a scale invariant binary MERA tensor network

    Args:
        device (str or None):       the device on which to run the simulation
        chi (int):                  bond dimension of the MERA
        numiter (int):              number of iteration steps 
        nsteps_steady_state (int):  number of power-method iteration steps for calculating the 
                                    steady state density matrices 
        verbose (int):              verbosity flag, if `verbose>0`, print out info  during optimization
        opt_u_after (int):          start optimizing disentangler only after `opt_u_after` initial optimization steps
        E_exact (float):            the exact ground-state energy (if known); default is the ground-state energy  of teh  
                                    infinite transverse field Ising model
    Returns: 
        wC (list of tf.Tensor):     optimized MERA isometries
        uC (list of tf.Tensor):     optimized MERA disentanglers
        rho (tf.Tensor):            steady state density matrices at the top layer 
        run_times (list of float):  run times per iteration step 
        Energies (list of float):   energies at each iteration step
    """
    
    graph = tf.Graph()


    with graph.as_default():
        with tf.device(device):
            #placeholders for feeding MERA tensors
            w_ph = tf.placeholder(dtype=dtype, shape=[None,None,None],name='w')
            u_ph=tf.placeholder(dtype=dtype, shape=[None,None,None,None],name='u')
            ham_ph=tf.placeholder(dtype=dtype, shape=[None,None,None,None,None,None],name='ham')
            rho_ph=tf.placeholder(dtype=dtype, shape=[None,None,None,None,None,None],name='rho')
            
            
            wC_op, uC_op, rhos_op = bml.initialize_binary_MERA_identities(2,chi,dtype)#initialization
            ham_op = bml.initialize_TFI_hams(dtype=dtype)
            
            ascend_ham_op = bml.ascending_super_operator(ham_ph, w_ph, u_ph) #ascending operation
            descend_rho_op = bml.descending_super_operator(rho_ph, w_ph, u_ph)#descending operation
            rho_ss_op = bml.steady_state_density_matrix(nsteps_steady_state, rho_ph,w_ph, u_ph)#steady-state operation
            
            u_env_op = bml.get_env_disentangler(ham_ph, rho_ph, w_ph, u_ph)
            w_env_op = bml.get_env_isometry(ham_ph, rho_ph, w_ph, u_ph)
            
            init_op = tf.global_variables_initializer()#initializer (not really needed unless random init is called)

            
    Energies = []
    run_times = {'env_u' : [], 'env_w' : [],
                 'steady_state' : [], 'svd_env_u' : [], 'svd_env_w' : [],
                 'ascend' : [], 'descend' : [], 'total' : []}
    
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_op)
        wC, uC, rho_0 = sess.run([wC_op,uC_op, rhos_op[-1]])
        ham = sess.run(ham_op)
        hams = [0 for x in range(len(wC) + 1)]
        rhos = [0 for x in range(len(wC) + 1)]

        hams[0] = ham
        chi1 = ham[0].shape[0]
        bias = np.max(np.linalg.eigvalsh(np.reshape(hams[0], (chi1 * chi1 * chi1, chi1 * chi1 * chi1)))) / 2
        hams[0] = hams[0] - bias * np.reshape(
           np.eye(chi1 * chi1 * chi1),(chi1, chi1, chi1, chi1, chi1, chi1))
    
        for k in range(numiter):
            if (verbose == 1) and (np.mod(k, 10) == 1):
                Z = ncon.ncon([rhos[0]],[1,2,3,1,2,3])
                energy = ncon.ncon([hams[0],rhos[0]],[[1,2,3,4,5,6],[4,5,6,1,2,3]])
                Energies.append(energy / Z + bias)
                stdout.write(
                    '\r     Iteration: %i of %i: E = %.8f, err = %.16f at D = %i with %i layers'
                    % (int(k), int(numiter), float(Energies[-1]),
                       float(Energies[-1] - E_exact), int(np.shape(wC[-1])[2]),
                       len(wC)))
                stdout.flush()
            t1 = time.time()            
            #obtain the steady state rho
            t2 = time.time()
            rho_0 = sess.run(rho_ss_op,feed_dict={rho_ph:rho_0, w_ph: wC[-1], u_ph : uC[-1]})
            run_times['steady_state'].append(time.time() - t2)
            
            rhos[-1] = rho_0      
            #get all other rhos
            t2 = time.time()
            for p in range(len(rhos) - 2, -1, -1):
                rhos[p] = sess.run(descend_rho_op,feed_dict = {rho_ph: rhos[p + 1], w_ph : wC[p], u_ph : uC[p]})
            run_times['descend'].append(time.time() - t2)                


            run_times['ascend'].append(0)
            run_times['svd_env_u'].append(0)
            run_times['svd_env_w'].append(0)
            run_times['env_u'].append(0)
            run_times['env_w'].append(0)
            
            for p in range(len(wC)): 
                #order of updates can be changed 
                if k >= opt_u_after:
                    
                    t2 = time.time()
                    uEnv = sess.run(u_env_op,feed_dict={ham_ph:hams[p],rho_ph:rhos[p+1], w_ph : wC[p], u_ph : uC[p]})
                    run_times['env_u'][-1] += (time.time() - t2)
                    
                    t2 = time.time()
                    uC[p] = u_update_svd(uEnv)
                    run_times['svd_env_u'][-1] += (time.time() - t2)

                t2 = time.time()                    
                wEnv = sess.run(w_env_op,feed_dict={ham_ph:hams[p],rho_ph:rhos[p+1], w_ph : wC[p], u_ph : uC[p]})
                run_times['env_w'][-1] += (time.time() - t2)
                
                t2 = time.time()                                    
                wC[p] = w_update_svd(wEnv)
                run_times['svd_env_w'][-1] += (time.time() - t2)

                t2 = time.time()                                    
                hams[p + 1] = sess.run(ascend_ham_op,feed_dict={ham_ph : hams[p], w_ph : wC[p], u_ph: uC[p]})
                run_times['ascend'][-1] += (time.time() - t2)
            run_times['total'].append(time.time() - t1)                
    return wC, uC, rhos[-1], run_times, Energies

def run_naive_optimization_benchmark(filename,
                                     chis,
                                     dtype=tf.float64,
                                     numiter=30,
                                     nsteps_steady_state=20,
                                     device=None):
    """
    run a naive optimization benchmark, i.e. one without growing bond dimensions by embedding 
    Args:
        filename (str):           filename under which results are stored as a pickle file
        chis (list):              list of bond dimensions for which to run the benchmark
        dtype (tensorflow dtype): dtype to be used for the benchmark
        numiter (int):            number of iteration steps 
        nsteps_steady_state (int):number of iterations steps used to obtain the steady-state 
                                  reduced density matrix
        device (str or None):     device on  which the benchmark should be run

    Returns: 
       dict:  dictionary containing the walltimes and energies
              key 'profile': list of runtimes
              key 'energies' list of energies per iteration step
    """

    walltimes = {'profile': {}, 'energies': {}}
    with tf.device(device):
        for chi in chis:
            print('running naive optimization benchmark for chi = {0}'.format(
                chi))
            wC, uC, rho_0, runtimes, energies = optimize_binary_mera_graphed_TFI(
                device=device,
                chi=chi,
                dtype=dtype,
                numiter=numiter,
                nsteps_steady_state=nsteps_steady_state,
                verbose=0,
                opt_u_after=0)

            walltimes['profile'][chi] = runtimes
            walltimes['energies'][chi] = energies
            print()
            print('runtimes, D={0}'.format(chi))
            print()
            for k, i in walltimes['profile'][chi].items():
                print(k, i)
            
            #print('     steps took {0} s'.format(walltimes['profile'][chi]))
            with open(filename + '.pickle', 'wb') as f:
                pickle.dump(walltimes, f)

    return walltimes



if __name__ == "__main__":
    """
    run benchmarks for a scale-invariant binary MERA optimization
    benchmark results are stored in disc
    """
    fname = 'binary_mera_benchmarks'
    if not os.path.exists(fname):
        os.mkdir(fname)
    os.chdir(fname)

    rootdir = os.getcwd()
    ######## comment out all benchmarks you don't want to run ########
    benchmarks = {
        'optimize_naive': {
            'chis': [4, 6],
            'dtype': tf.float64,
            'nsteps_steady_state': 10,
            'numiter': 40
        }
    }
    date = datetime.date
    today = str(date.today())
    use_gpu = False  #use True when running on GPU
    #list available devices
    DEVICES = tf.contrib.eager.list_devices()
    print("Available devices:")
    for i, device in enumerate(DEVICES):
        print("%d) %s" % (i, device))
    CPU = '/device:CPU:0'
    GPU = '/job:localhost/replica:0/task:0/device:GPU:0'
    if use_gpu:
        specified_device_type = GPU
        name = today + 'GPU'
    else:
        specified_device_type = CPU
        name = today + 'CPU'

    if 'optimize_naive' in benchmarks:
        filename = name + 'graphed_binary_mera_naive_optimization_benchmark_Nthreads{}'.format(
            NUM_THREADS)
        keys = sorted(benchmarks['optimize_naive'].keys())
        for key in keys:
            val = benchmarks['optimize_naive'][key]
            if hasattr(val, 'name'):
                val = val.name

            filename = filename + '_' + str(key) + str(val)
        filename = filename.replace(' ', '')

        fname = 'benchmarks_optimize_naive'
        if not os.path.exists(fname):
            os.mkdir(fname)

        os.chdir(fname)
        run_naive_optimization_benchmark(
            filename,
            **benchmarks['optimize_naive'],
            device=specified_device_type)
        os.chdir(rootdir)

