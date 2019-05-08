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

NUM_THREADS = 4
import os
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

import tensorflow as tf
import copy
import numpy as np
import time
import pickle
import lib.ncon as ncon
import binary_mera_lib_np as bml_np
import binary_mera_lib as bml

import misc_mera
from sys import stdout

config = tf.ConfigProto()
config.intra_op_parallelism_threads = NUM_THREADS
config.inter_op_parallelism_threads = 1

def optimize_binary_mera(sess,
                         ham_0,
                         wC,
                         uC,
                         isometry, unitary, rho, ham,
                         ascending_super_operator,
                         descending_super_operator,                         
                         steady_state_density_matrix,
                         get_env_isometry,                         
                         get_env_disentangler,
                         rho_0=None,
                         numiter=1000,
                         nsteps_steady_state=4,
                         verbose=0,
                         opt_u=True,
                         opt_w=True,
                         opt_u_after=40):
    """
    
    """
    dtype = ham_0.dtype
    
    hams = [0 for x in range(len(wC) + 1)]
    rhos = [0 for x in range(len(wC) + 1)]
    
    hams[0] = ham_0
    
    chi1 = hams[0].shape[0]
    
    bias = np.max(np.linalg.eigvalsh(np.reshape(hams[0],(chi1 * chi1 * chi1, chi1 * chi1 * chi1))))/2
    hams[0] = hams[0] - bias * np.reshape(np.eye(chi1 * chi1 * chi1).astype(dtype=dtype), (chi1, chi1, chi1, chi1, chi1, chi1))

    Energies = []
    run_times = []
    if not np.all(rho_0):
        chi_max = wC[-1].shape[2]
        rho_0 = np.reshape(np.eye(chi_max ** 3).astype(dtype=dtype),(chi_max, chi_max, chi_max, chi_max, chi_max, chi_max))
    for k in range(numiter):
        t1 = time.time()
        rho_0 = sess.run(steady_state_density_matrix, feed_dict={rho: rho_0, isometry: wC[-1], unitary: uC[-1]})
        rhos[-1] = rho_0
        for p in range(len(rhos)-2,-1,-1):
            rhos[p] = sess.run(descending_super_operator, feed_dict={rho : rhos[p+1], isometry: wC[p], unitary: uC[p]})
        
        if verbose > 0:
            if np.mod(k,10) == 1:
                Z = ncon.ncon([rhos[0]],[[1,2,3,1,2,3]])
                Energies.append(((ncon.ncon([rhos[0],hams[0]],[[1, 2, 3, 4, 5, 6],[1, 2, 3, 4, 5, 6]])) + bias)/Z)
                stdout.write('\r     Iteration: %i of %i: E = %.8f, err = %.16f at D = %i with %i layers' %
                             (int(k),int(numiter), float(Energies[-1]), float(Energies[-1] + 4/np.pi,), int(wC[-1].shape[2]), len(wC)))
                stdout.flush()
                
        for p in range(len(wC)):
            if k >= opt_u_after:
                uEnv = sess.run(get_env_disentangler,feed_dict={ham : hams[p], rho : rhos[p+1], isometry : wC[p], unitary : uC[p]})
                if opt_u:
                    uC[p] = bml_np.u_update_svd_numpy(uEnv)

            wEnv = sess.run(get_env_isometry,feed_dict={ham : hams[p], rho : rhos[p+1], isometry : wC[p], unitary : uC[p]})
            if opt_w:
                wC[p] = bml_np.w_update_svd_numpy(wEnv)
                        
            hams[p+1] = sess.run(ascending_super_operator, feed_dict={ham : hams[p], isometry : wC[p], unitary : uC[p]})
            
        run_times.append(time.time() - t1)
        if verbose > 2:
            print('time per iteration: ',run_times[-1])
            
    return wC, uC, rhos[-1], run_times, Energies



def run_binary_mera_optimization_TFI(sess, device,
                                     chis=[4, 6, 8],
                                     niters=[200, 300, 1000],
                                     embedding=None,
                                     dtype=tf.float64,
                                     verbose=1,
                                     nsteps_steady_state=4,
                                     opt_u_after=40,
                                     wC=0,
                                     uC=0,
                                     rho_0=0):
    
    """
    binary mera optimization
    Parameters:
    -------------------
    sess:                 tf.Session object
    device:               a dervice on which to run the optimization
    chis:                 list of int 
                          bond dimension of successive MERA simulations 
    niters:               list of int 
                          number of optimization steps of successive MERA optimizations 
    embedding:            list of str or None
                          type of embedding scheme used to embed mera into the next larger bond dimension 
                          entries can be: 'p' or 'pad' for padding with zeros without, if possible, adding new layers 
                                          'a' or 'add' for adding new layer with increased  bond dimension
    dtype:                tensorflow dtype 
    verbose:              int 
                          verbosity flag 
    nsteps_steady_state:  int 
                          number power iteration of steps used to obtain the steady state reduced 
                          density matrix

    Returns: 
    --------------------
    (energies, walltimes, wC, uC)
    energies:   list of tf.Tensor of shape () 
                energies at iterations steps
    walltimes:  list of float 
                walltimes per iteration step 
    wC, uC:     list of tf.Tensor
                isometries (wC) and disentanglers (uC)
    """
    #initialize all placeholders
    with tf.device(device):
        isometry = tf.placeholder(dtype, shape = [None, None, None], name='isometry')
        unitary = tf.placeholder(dtype, shape = [None, None, None, None], name = 'unitary')
        rho = tf.placeholder(dtype, shape = [None, None, None, None, None, None], name='rho')
        ham = tf.placeholder(dtype, shape = [None, None, None, None, None, None], name='ham')
        rho_ss = bml.steady_state_density_matrix(nsteps_steady_state, rho, isometry, unitary)
        desc = bml.descending_super_operator(rho, isometry, unitary)
        asc = bml.ascending_super_operator(ham, isometry, unitary)
        uenv = bml.get_env_disentangler(ham, rho, isometry, unitary)
        wenv = bml.get_env_isometry(ham, rho, isometry, unitary)

    if not embedding:
        embedding = ['p']*len(chis)

    if wC == 0:
        wC, _, _ = bml_np.initialize_binary_MERA(phys_dim=2, chi=chis[0],dtype=dtype.as_numpy_dtype)
    if uC == 0:
        _, uC, _ = bml_np.initialize_binary_MERA(phys_dim=2, chi=chis[0],dtype=dtype.as_numpy_dtype)
    if rho_0 == 0:
        _, _, rho_0 = bml_np.initialize_binary_MERA(phys_dim=2, chi=chis[0],dtype=dtype.as_numpy_dtype)
        
    ham_0 = bml_np.initialize_TFI_hams(dtype=dtype.as_numpy_dtype)
    energies = []
    walltimes = []
    init = True
    for chi, niter, which in zip(chis, niters, embedding):
        if not init:
            if which in ('a','add'):
                wC, uC = bml_np.unlock_layer(wC, uC, noise=noise)
                wC, uC = bml_np.pad_mera_tensors(chi, wC, uC, noise=noise)
            elif which in ('p','pad'):        
                wC, uC = bml_np.pad_mera_tensors(chi, wC, uC, noise=noise)

            
        rho_0 = bml_np.pad_tensor(rho_0, [chi, chi, chi, chi, chi, chi])
        wC, uC, rho_0, times, es = optimize_binary_mera(sess=sess,
                                                        ham_0=ham_0,
                                                        wC=wC, uC=uC,
                                                        isometry=isometry,
                                                        unitary=unitary,
                                                        ham=ham,
                                                        rho=rho,
                                                        ascending_super_operator=asc,
                                                        descending_super_operator=desc,
                                                        steady_state_density_matrix=rho_ss,
                                                        get_env_isometry=wenv,                                                        
                                                        get_env_disentangler=uenv,
                                                        rho_0=rho_0,                                                        
                                                        numiter=niter,
                                                        nsteps_steady_state=nsteps_steady_state,
                                                        verbose=verbose,
                                                        opt_u=True,
                                                        opt_w=True,
                                                        opt_u_after=opt_u_after)
        energies.extend(es)
        walltimes.extend(times)
        init = False
    return wC, uC, walltimes, energies

def run_naive_optimization_benchmark(sess,
                                     device,
                                     filename,
                                     chis=[4, 6, 8, 10, 12],
                                     dtype=tf.float64,
                                     numiter=10,
                                     nsteps_steady_state=4,
                                     opt_u=True,
                                     opt_w=True,
                                     opt_u_after=40):

    walltimes = {'profile': {}, 'energies' : {}}    
    with tf.device(device):        
        for chi in chis:
            print('running naive optimization benchmark for chi = {0}'.
                  format(chi))


            wC, uC, runtimes, energies = run_binary_mera_optimization_TFI(sess=sess, device=device,
                                                                          chis=[chi],
                                                                          niters=[numiter],
                                                                          dtype=dtype,
                                                                          verbose=1,
                                                                          nsteps_steady_state=nsteps_steady_state,
                                                                          opt_u_after=opt_u_after)

                

            walltimes['profile'][chi] = runtimes
            walltimes['energies'][chi] = energies
            print()
            print('     steps took {0} s'.format(walltimes['profile'][chi]))
            with open(filename + '.pickle', 'wb') as f:
                pickle.dump(walltimes, f)

    return walltimes



if __name__ == "__main__":
    fname =  'binary_mera_benchmarks'            
    if not os.path.exists(fname):
        os.mkdir(fname)
    os.chdir(fname)
    
    rootdir = os.getcwd()
    benchmarks = {'optimize_naive' : {'chis' :  [4, 6],
                                      'dtype' : tf.float64,
                                      'opt_u' : True,
                                      'opt_w' : True,
                                      'numiter' : 5}}

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

        
    if 'optimize_naive' in benchmarks:
        filename = name + 'binary_mera_naive_optimization_benchmark_Nthreads{}'.format(NUM_THREADS)
        keys = sorted(benchmarks['optimize_naive'].keys())
        for key in keys:
            val = benchmarks['optimize_naive'][key]
            if hasattr(val, 'name'):                                
                val = val.name
            
            filename = filename + '_' + str(key) + str(val)
        filename = filename.replace(' ', '')
        
        fname = 'benchmarks_optimize_naive_feed_dict'
        if not os.path.exists(fname):
            os.mkdir(fname)
            
        os.chdir(fname)
        with tf.Session(config=config) as sess:
            run_naive_optimization_benchmark(sess=sess,
                                             device=specified_device_type,
                                             filename=filename,
                                             **benchmarks['optimize_naive'],
                                             opt_u_after=0)
        os.chdir(rootdir)            
            
