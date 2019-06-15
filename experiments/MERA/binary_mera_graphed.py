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
    phys_dim = 2
    with graph.as_default():
        with tf.device(device):
            ###################################################################################################                
            #define variables for all objects
            #we use more Variables than usually neccessary 
            #the reason is that we want to time certain operations
            wC_op, uC_op, rhos_op = bml.initialize_binary_MERA_identities(phys_dim,chi,dtype)#initialization
            wC =  [tf.Variable(wC_op[n],name='wC_{0}'.format(n)) for n in range(len(wC_op))] 
            uC =  [tf.Variable(uC_op[n], name = 'uC_{0}'.format(n)) for n in range(len(uC_op))]
            u_envs = [tf.Variable(uC_op[n], name = 'u_env_{0}'.format(n)) for n in range(len(uC_op))]
            w_envs = [tf.Variable(wC_op[n], name = 'w_env_{0}'.format(n)) for n in range(len(wC_op))]
            u_updates = [tf.Variable(uC_op[n], name = 'u_update_{0}'.format(n)) for n in range(len(uC_op))]
            w_updates = [tf.Variable(wC_op[n], name = 'w_update_{0}'.format(n)) for n in range(len(wC_op))]
    
            rhos = [tf.Variable(rhos_op[n], name='rho_{0}'.format(n)) for n in range(len(rhos_op))]
            hams = [bml.initialize_TFI_hams(dtype=dtype)] 
            chi1 = tf.shape(hams[0])[0]
            bias = tf.Variable(tf.cast(tf.math.reduce_max(
                    tf.cast(tf.linalg.eigvalsh(
                        tf.reshape(hams[0], (chi1 * chi1 * chi1, chi1 * chi1 * chi1))),dtype)) / 2,dtype),name='bias')
    
            hams[0] = tf.Variable(hams[0] - bias.initialized_value() * tf.reshape(
                tf.eye(chi1 * chi1 * chi1, dtype=dtype),
                (chi1, chi1, chi1, chi1, chi1, chi1)), name='ham_0')
    
            for n in range(len(wC)):
                hams.append(tf.Variable(bml.ascending_super_operator(hams[-1].initialized_value(), 
                                                                     wC[n].initialized_value(), 
                                                                     uC[n].initialized_value()),
                                            name='ham_{0}'.format(n+1)))
            ############################    done creating variables     #######################################
            
            #an op for checking the energy   
            energy = tn.ncon([hams[0].initialized_value(),rhos[0].initialized_value()],
                     [[1,2,3,4,5,6],[4,5,6,1,2,3]])/tn.ncon([rhos[0].initialized_value()],[[1,2,3,1,2,3]])+\
                    bias.initialized_value()
            
            init_op = tf.global_variables_initializer()
    
            descend_ops = [bml.descending_super_operator(rhos[p+1], wC[p], uC[p])
                           for p in range(len(rhos)-1)]
            ascend_ops = [bml.ascending_super_operator(hams[p], wC[p], uC[p])
                          for p in range(len(hams)-1)]
           
            rho_ss_op = bml.steady_state_density_matrix(nsteps_steady_state, rhos[-1],wC[-1], uC[-1])
            assign_rhos = [tf.assign(rhos[n], descend_ops[n]) for n in range(len(rhos)-1)]
            assign_rhos.append(tf.assign(rhos[-1], rho_ss_op))
            assign_hams = [hams[0]] + [tf.assign(hams[n+1], ascend_ops[n]) for n in range(len(hams)-1)]
            assign_u_envs = [tf.assign(u_envs[p], bml.get_env_disentangler(hams[p], rhos[p+1], wC[p], uC[p]))
                               for p in range(len(uC))]
            assign_w_envs = [tf.assign(w_envs[p], bml.get_env_isometry(hams[p], rhos[p+1], wC[p], uC[p]))
                                      for p in range(len(wC))]
            assign_u_updates = [tf.assign(u_updates[n], misc_mera.u_update_svd(u_envs[n]))
                                for n in range(len(uC))]
            assign_w_updates = [tf.assign(w_updates[n], misc_mera.w_update_svd(w_envs[n]))
                                for n in range(len(w_envs))]
            assign_us = [tf.assign(uC[n], u_updates[n]) for n in range(len(uC))]
            assign_ws = [tf.assign(wC[n], w_updates[n]) for n in range(len(wC))]
            
    with tf.Session(graph=graph,config=config) as sess:
        Energies = []
        run_times = {'env_u' : [], 'env_w' : [],
                     'steady_state' : [], 'svd_env_u' : [], 'svd_env_w' : [],
                     'ascend' : [], 'descend' : [], 'total' : []}
        
        sess.run(init_op)
        for k in range(numiter):
            if (k % 10 == 0) and (verbose == 1):
                
                Energies.append(sess.run(energy))
                stdout.write(
                    '\r     Iteration: %i of %i: E = %.8f, err = %.16f'
                    % (int(k), int(numiter), float(Energies[-1]),
                       float(Energies[-1] - E_exact)))
                stdout.flush()

                
            run_times['descend'].append(0)
            run_times['ascend'].append(0)
            run_times['svd_env_u'].append(0)
            run_times['svd_env_w'].append(0)
            run_times['env_u'].append(0)
            run_times['env_w'].append(0)
                
            tinit = time.time()            
            #obtain the steady state rho
            t2 =  time.time()
            sess.run(assign_rhos[-1].op)
            run_times['steady_state'].append(time.time() - t2)
            
                        
            for p in reversed(range(len(rhos) - 1)):
                t2 =  time.time()                
                sess.run(assign_rhos[p].op)
                run_times['descend'][-1] += (time.time() - t2)                  

            for p in range(len(wC)): 
                #order of updates can be changed
                
                if k >= opt_u_after:
                    t2 =  time.time()                
                    sess.run(assign_u_envs[p].op)    #obtain environment and assign to a Variable
                    run_times['env_u'][-1] += (time.time() - t2)

                    t2 =  time.time()                
                    sess.run(assign_u_updates[p].op) #obtain update by svd of the env and assign to a Variable
                    run_times['svd_env_u'][-1] += (time.time() - t2)
                    
                    sess.run(assign_us[p].op)        #assign the calculated update to the uC[p] tf.Variable

                t2 =  time.time()
                sess.run(assign_w_envs[p].op)        #obtain environment and assign to the variable
                run_times['env_w'][-1] += (time.time() - t2)

                t2 =  time.time()
                sess.run(assign_w_updates[p].op)     #obtain update by svd of the env and assign to a Variable
                run_times['svd_env_w'][-1] += (time.time() - t2)
                
                sess.run(assign_ws[p].op)            #assign the calculated update to the wC[p] tf.Variable
                
                t2 =  time.time()                
                sess.run(assign_hams[p + 1].op)      #calculate and assign the new Hamiltonians in layer p+1
                run_times['ascend'][-1] += (time.time() - t2)                                
            run_times['total'].append(time.time() - tinit)
            
    return wC, uC, rhos[-1], run_times, Energies            

def run_naive_optimization_benchmark(filename,
                                     chis,
                                     dtype=tf.float64,
                                     numiter=30,
                                     nsteps_steady_state=10,
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
                verbose=1,
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
            'chis': [4, 6,  8, 10, 12, 14, 16],
            'dtype': tf.float64,
            'nsteps_steady_state': 10,
            'numiter': 40
        }
    }
    date = datetime.date
    today = str(date.today())
    use_gpu = False #use True when running on GPU
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

p
