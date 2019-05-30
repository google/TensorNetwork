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
import tensornetwork as tn
import experiments.MERA.binary_mera_lib as bml
import experiments.MERA.misc_mera as misc_mera
from sys import stdout
config = tf.ConfigProto()
config.intra_op_parallelism_threads = NUM_THREADS
config.inter_op_parallelism_threads = 1
tf.enable_eager_execution(config)
tf.enable_v2_behavior()


def run_binary_mera_optimization_TFI(chis=[4, 6, 8],
                                     niters=[200, 300, 1000],
                                     embeddings=None,
                                     dtype=tf.float64,
                                     verbose=1,
                                     nsteps_steady_state=4,
                                     numpy_update=True,
                                     opt_u_after=40,
                                     opt_all_layers=None,
                                     wC=0,
                                     uC=0,
                                     rho_0=0,
                                     noises=None,
                                     filename=None):
    """
    optimize a binary mera to approximate the ground-state of the infinite transverse field Ising model
    Args:
        chis (list of int):   bond dimension of successive MERA simulations 
        niters (list of int): number of optimization steps of successive MERA optimizations 
        embeddings (list of str or None): type of embedding scheme used to embed mera into the next larger bond dimension 
                                          entries can be: 'p' or 'pad' for padding with zeros without, if possible, adding new layers 
                                                          'a' or 'add' for adding new layer with increased  bond dimension
                                                          'n'          for keeping the MERA as it is (e.g. for resuming optimization)
                                          the first entry will be ignored for the case where no `wC` and `uC` tensors are passed
        dtype (tensorflow dtype):      dtype
        verbose (int):                 verbosity flag, if `verbose>0`, print out info  during optimization
        nsteps_steady_state (int):     number of power-method iteration steps for calculating the 
                                       steady state density matrices 
        numpy_update (bool):           if True, use numpy svd to calculate update of disentanglers and isometries
        opt_u_after (int):             start optimizing disentangler only after `opt_u_after` initial optimization steps
        opt_all_layers (bool):         if `True`, optimize all layers
                                       if `False`, only optimize truncating layers
        wC (list of tf.Tensor or 0.0): initial values for isometries; if `0.0`, initialize with  identities
        uC (list of tf.Tensor or 0.0): initial values for disentanglers; if `0.0`, initialize with  identities
        rho_0 (tf.Tensor or 0.0):      initial value for reduced density matrix; if `0.0`, initialize with  identities
        noises (list of float):        noise values for initializing new layers

    Returns: 
        wC (list of tf.Tensor): optimized isometries of the MERA
        uC (list of tf.Tensor): optimized disentanglers of the MERA
        energies (list of tf.Tensor): energies per iteration steps
        walltimes (list of float):    walltimes per iteration step 
    """

    if not embeddings:
        embeddings = ['p'] * len(chis)
    if not noises:
        noises = [0.0] * len(chis)
    if not opt_all_layers:
        opt_all_layers = [True] * len(chis)

    init = False
    if wC == 0:
        init = True
        wC, _, _ = bml.initialize_binary_MERA_identities(
            phys_dim=2, chi=chis[0], dtype=dtype)
    if uC == 0:
        init = True
        _, uC, _ = bml.initialize_binary_MERA_identities(
            phys_dim=2, chi=chis[0], dtype=dtype)
    if rho_0 == 0:
        _, _, rho_0 = bml.initialize_binary_MERA_identities(
            phys_dim=2, chi=chis[0], dtype=dtype)

    ham_0 = bml.initialize_TFI_hams(dtype=dtype)

    data = {'profile': {}, 'energies': {}}

    for chi, niter, which, noise, opt_all in zip(chis, niters, embeddings,
                                                 noises, opt_all_layers):
        energies = []
        walltimes = []
        if not init:
            if which in ('a', 'add'):
                wC, uC = bml.unlock_layer(wC, uC, noise=noise)
                wC, uC = bml.pad_mera_tensors(chi, wC, uC, noise=noise)
            elif which in ('p', 'pad'):
                wC, uC = bml.pad_mera_tensors(chi, wC, uC, noise=noise)

        rho_0 = misc_mera.pad_tensor(rho_0, [chi, chi, chi, chi, chi, chi])

        wC, uC, rho_0, times, es = bml.optimize_binary_mera(
            ham_0=ham_0,
            #rho_0=rho_0,
            wC=wC,
            uC=uC,
            numiter=niter,
            nsteps_steady_state=nsteps_steady_state,
            verbose=verbose,
            opt_u=True,
            opt_w=True,
            numpy_update=numpy_update,
            opt_u_after=opt_u_after,
            opt_all_layers=opt_all)
        energies.extend(es)
        walltimes.extend(times)
        data['profile'][chi] = walltimes
        data['energies'][chi] = energies
        init = False
        if filename:
            with open(filename + '_tensors.pickle', 'wb') as f:
                pickle.dump([wC, uC], f)
            with open('energies_walltimes_' + filename + '.pickle', 'wb') as f:
                pickle.dump(data, f)

    return wC, uC, walltimes, energies


def benchmark_ascending_operator(ham, w, u, num_layers):
    """
    run benchmark for ascending super operator
    Args: 
        rhoab (tf.Tensor):  reduced densit matrix on a-b lattice
        rhoba (tf.Tensor):  reduced densit matrix on b-a lattice
        w   (tf.Tensor):  isometry
        v   (tf.Tensor):  isometry
        u   (tf.Tensor):  disentangler
        num_layers(int):  number of layers over which to descend the hamiltonian
    Returns:
        runtime (float):  the runtime
    """
    t1 = time.time()
    for t in range(num_layers):
        ham = bml.ascending_super_operator(ham, w, u)
    return time.time() - t1


def benchmark_descending_operator(rho, w, u, num_layers):
    """
    run benchmark for descending super operator
    Args: 
        rhoab (tf.Tensor):  reduced densit matrix on a-b lattice
        rhoba (tf.Tensor):  reduced densit matrix on b-a lattice
        w   (tf.Tensor):  isometry
        v   (tf.Tensor):  isometry
        u   (tf.Tensor):  disentangler
        num_layers(int):  number of layers over which to descend the hamiltonian
    Returns:
        runtime (float):  the runtime
    """

    t1 = time.time()
    for p in range(num_layers):
        rho = bml.descending_super_operator(rho, w, u)
    return time.time() - t1


def run_ascending_operator_benchmark(filename,
                                     chis=[4, 6, 8, 10, 12],
                                     num_layers=1,
                                     dtype=tf.float64,
                                     device=None):
    """
    run ascending operators benchmarks and save benchmark data in `filename`
    Args:
        filename (str):  filename under which results are stored as a pickle file
        chis (list):  list of bond dimensions for which to run the benchmark
        num_layers (int): number of layers over which to ascend an operator 
        dtype (tensorflow dtype): dtype to be used for the benchmark
        device (str):             device on  which the benchmark should be run
    Returns: 
       dict:  dictionary containing the walltimes
              key 'warmup' contains warmup (i.e. first run) runtimes
              key 'profile' contains subsequent runtimes
    """

    walltimes = {'warmup': {}, 'profile': {}}
    for chi in chis:
        print('running ascending-operator benchmark for chi = {0} benchmark'.
              format(chi))
        with tf.device(device):
            w = tf.random_uniform(shape=[chi, chi, chi], dtype=dtype)
            u = tf.random_uniform(shape=[chi, chi, chi, chi], dtype=dtype)
            ham = tf.random_uniform(
                shape=[chi, chi, chi, chi, chi, chi], dtype=dtype)
            walltimes['warmup'][chi] = benchmark_ascending_operator(
                ham, w, u, num_layers)
            print('     warmup took {0} s'.format(walltimes['warmup'][chi]))
            walltimes['profile'][chi] = benchmark_ascending_operator(
                ham, w, u, num_layers)
            print('     profile took {0} s'.format(walltimes['profile'][chi]))

    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(walltimes, f)
    return walltimes


def run_descending_operator_benchmark(filename,
                                      chis=[4, 6, 8, 10, 12],
                                      num_layers=1,
                                      dtype=tf.float64,
                                      device=None):
    """
    run descending operators benchmarks and save benchmark data in `filename`
    Args:
        filename (str):  filename under which results are stored as a pickle file
        chis (list):  list of bond dimensions for which to run the benchmark
        num_layers (int): number of layers over which to descend the reduced density matrix
        dtype (tensorflow dtype): dtype to be used for the benchmark
        device (str):             device on  which the benchmark should be run
    Returns: 
       dict:  dictionary containing the walltimes
              key 'warmup' contains warmup (i.e. first run) runtimes
              key 'profile' contains subsequent runtimes
    """

    walltimes = {'warmup': {}, 'profile': {}}
    for chi in chis:
        print('running descending-operator benchmark for chi = {0} benchmark'.
              format(chi))
        with tf.device(device):
            w = tf.random_uniform(shape=[chi, chi, chi], dtype=dtype)
            u = tf.random_uniform(shape=[chi, chi, chi, chi], dtype=dtype)
            rho = tf.random_uniform(
                shape=[chi, chi, chi, chi, chi, chi], dtype=dtype)
            walltimes['warmup'][chi] = benchmark_descending_operator(
                rho, w, u, num_layers=num_layers)
            print('     warmup took {0} s'.format(walltimes['warmup'][chi]))
            walltimes['profile'][chi] = benchmark_descending_operator(
                rho, w, u, num_layers=num_layers)
            print('     profile took {0} s'.format(walltimes['profile'][chi]))

    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(walltimes, f)
    return walltimes


def run_naive_optimization_benchmark(filename,
                                     chis=[4, 6, 8, 10, 12],
                                     dtype=tf.float64,
                                     numiter=10,
                                     nsteps_steady_state=4,
                                     opt_u=True,
                                     opt_w=True,
                                     numpy_update=True,
                                     device=None,
                                     opt_u_after=40):
    """
    run a naive optimization benchmark, i.e. one without growing bond dimensions by embedding 
    Args:
        filename (str):           filename under which results are stored as a pickle file
        chis (list):              list of bond dimensions for which to run the benchmark
        dtype (tensorflow dtype): dtype to be used for the benchmark
        numiter (int):            number of iteration steps 
        nsteps_steady_state (int):number of iterations steps used to obtain the steady-state 
                                  reduced density matrix
        opt_u (bool):             if True, optimize disentangler `u`
        opt_w (bool):             if True, optimize isometries `w`
        numpy_update (bool):      if True, use numpy-svd to update tensors
        device (str or None):     device on  which the benchmark should be run
        opt_u_after (int):        do not optimize `u` for the first `opt_u_after' steps

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

            wC, uC, rho_0 = bml.initialize_binary_MERA_identities(
                phys_dim=2, chi=chi, dtype=dtype)
            ham_0 = bml.initialize_TFI_hams(dtype=dtype)
            wC, uC, rho_0, runtimes, energies = bml.optimize_binary_mera(
                ham_0=ham_0,
                rho_0=rho_0,
                wC=wC,
                uC=uC,
                numiter=numiter,
                nsteps_steady_state=nsteps_steady_state,
                verbose=1,
                opt_u=opt_u,
                opt_w=opt_w,
                numpy_update=numpy_update,
                opt_u_after=opt_u_after)

            walltimes['profile'][chi] = runtimes
            walltimes['energies'][chi] = energies
            print('     steps took {0} s'.format(walltimes['profile'][chi]))
            with open(filename + '.pickle', 'wb') as f:
                pickle.dump(walltimes, f)

    return walltimes


def run_optimization_benchmark(filename,
                               chis=[4, 8, 10],
                               numiters=[200, 200, 400],
                               embeddings=None,
                               dtype=tf.float64,
                               verbose=1,
                               opt_u_after=10,
                               nsteps_steady_state=4,
                               device=None):
    """
    run a realistic optimization benchmark, i.e. one with growing bond dimensions by embedding 
    Args:
        filename (str):            filename under which results are stored as a pickle file
        chis (list):               list of bond dimensions. optimization starts with chi[0], then 
                                   sequentially increases the bond dimension as given chis[n] for n > 0
        numiters (list):           maximum number of iteration steps per bond dimension
        embeddings (list or None): list of str of len(chis). elements can be either 'a' or 'p'.
                                   if embeddings[n]='a': embed the mera from iteration n - 1 by adding 
                                                         a new  layer of mera-tensors of bond  dimension `chi[n]`
                                   if embeddings[n]='p': embed the mera from iteration n - 1 by padding tensors
                                                         with zeros to dimension `chis[n]`; if `chis[n]`
                                                         can't be obtained from padding, pad tensors to thir maximal dimension
                                                         and a new layer.
        dtype (tensorflow dtype):  dtype to be used for the benchmark
        verbose (int):             verbosity flag; if `verbose > 0`, print out info during simulation
        opt_u_after (int):         do not optimize `u` for the first `opt_u_after' steps
        nsteps_steady_state (int): number of iterations steps used to obtain the steady-state 
                                   reduced density matrix
        device (str):              device on  which the benchmark should be run

    Returns: 
       dict:  dictionary containing the walltimes and energies
              key 'profile': list of runtimes
              key 'energies' list of energies per iteration step

    """

    walltimes = {}
    with tf.device(device):
        print('     running optimization benchmark')

        wC, uC, runtimes, energies = run_binary_mera_optimization_TFI(
            chis=chis,
            niters=numiters,
            embeddings=embeddings,
            dtype=dtype,
            verbose=verbose,
            nsteps_steady_state=nsteps_steady_state,
            opt_u_after=opt_u_after,
            filename=filename,
            numpy_update=True)

        walltimes['profile'] = runtimes
        walltimes['energies'] = energies
        with open(filename + '.pickle', 'wb') as f:
            pickle.dump(walltimes, f)
        with open(filename + '_tensors.pickle', 'wb') as f:
            print('saving to', filename)
            pickle.dump([wC, uC], f)

    return walltimes


def test_ascending_descending(chi=4, dtype=tf.float64):
    """
    test if ascending and descending operations are doing the right thing
    """
    wC, uC, rho_0 = bml.initialize_binary_MERA_identities(
        phys_dim=2, chi=4, dtype=dtype)
    for n in range(5):
        wC.append(copy.copy(wC[-1]))
        uC.append(copy.copy(uC[-1]))
    wC, uC, _, _ = run_binary_mera_optimization_TFI(
        chis=[chi],
        niters=[10],
        opt_u_after=0,
        embeddings=['a'],
        dtype=dtype,
        wC=wC,
        uC=uC)
    ham_0 = bml.initialize_TFI_hams(dtype)
    rho = [0 for n in range(len(wC) + 1)]
    ham = [0 for n in range(len(wC) + 1)]
    rho[-1] = bml.steady_state_density_matrix(10, rho_0, wC[-1], uC[-1])
    ham[0] = ham_0
    print()
    for p in range(len(rho) - 2, -1, -1):
        rho[p] = bml.descending_super_operator(rho[p + 1], wC[p], uC[p])
    for p in range(len(wC)):
        ham[p + 1] = bml.ascending_super_operator(ham[p], wC[p], uC[p])
    energies = [
        tn.ncon([rho[p], ham[p]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
        for p in range(len(rho))
    ]
    print('following numbers should all be 1/2')
    print(
        np.array(
            [energies[p] / energies[p + 1] for p in range(len(energies) - 1)]))


if __name__ == "__main__":
    """
    run benchmarks for a scale-invariant binary MERA optimization
    benchmark results are stored in disc
    """
    if not tf.executing_eagerly():
        pass

    else:
        fname = 'binary_mera_benchmarks'
        if not os.path.exists(fname):
            os.mkdir(fname)
        os.chdir(fname)

        rootdir = os.getcwd()
        ######## comment out all benchmarks you don't want to run ########
        benchmarks = {
            'ascend': {
                'chis': [4, 6, 8],
                'dtype': tf.float64,
                'num_layers': 1
            },
            'descend': {
                'chis': [4, 6, 8],
                'dtype': tf.float64,
                'num_layers': 1
            },
            'optimize_naive': {
                'chis': [4, 6, 8],
                'dtype': tf.float64,
                'opt_u': True,
                'opt_w': True,
                'numpy_update': True,
                'nsteps_steady_state': 10,
                'numiter': 2
            },
            'optimize': {
                'chis': [4, 6, 8],
                'numiters': [400, 400, 400],
                'embeddings': ['p', 'a', 'p'],
                'nsteps_steady_state': 10,
                'opt_u_after': 20,
                'dtype': tf.float64
            }
        }

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
            name = 'GPU'
        else:
            specified_device_type = CPU
            name = 'CPU'

        if 'ascend' in benchmarks:
            filename = name + 'binary_mera_ascending_benchmark'
            for key, val in benchmarks['ascend'].items():
                if hasattr(val, 'name'):
                    val = val.name
                filename = filename + '_' + str(key) + str(val)
            filename = filename.replace(' ', '')

            fname = 'ascending_benchmarks'
            if not os.path.exists(fname):
                os.mkdir(fname)
            os.chdir(fname)
            run_ascending_operator_benchmark(
                filename, device=specified_device_type, **benchmarks['ascend'])
            os.chdir(rootdir)

        if 'descend' in benchmarks:
            filename = name + 'binary_mera_descending_benchmark'
            for key, val in benchmarks['descend'].items():
                if hasattr(val, 'name'):
                    val = val.name

                filename = filename + '_' + str(key) + str(val)
            filename = filename.replace(' ', '')

            fname = 'descending_benchmarks'
            if not os.path.exists(fname):
                os.mkdir(fname)
            os.chdir(fname)

            run_descending_operator_benchmark(
                filename, device=specified_device_type, **benchmarks['descend'])
            os.chdir(rootdir)

        if 'optimize_naive' in benchmarks:
            filename = name + 'binary_mera_naive_optimization_benchmark_Nthreads{}'.format(
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
                opt_u_after=0,
                device=specified_device_type)
            os.chdir(rootdir)

        if 'optimize' in benchmarks:
            filename = name + 'binary_mera_optimization_benchmark_Nthreads{}'.format(
                NUM_THREADS)
            fname = 'benchmarks_optimize'
            if not os.path.exists(fname):
                os.mkdir(fname)
            os.chdir(fname)

            for key, val in benchmarks['optimize'].items():
                if hasattr(val, 'name'):
                    val = val.name
                filename = filename + '_' + str(key) + str(val)
            filename = filename.replace(' ', '')
            run_optimization_benchmark(
                filename,
                device=specified_device_type,
                **benchmarks['optimize'])

            os.chdir(rootdir)
