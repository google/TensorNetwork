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


"""implementation Lanczos algorithms for tridiagonalization of hermitian sparse matrices."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../')
from tensorflow.contrib.solvers.python.ops import util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

import ncon as ncon
import numpy as np
import time
import tensorflow as tf
import collections
import copy

@tf.contrib.eager.defun
def do_lanczos_compiled(L, mpo, R, initial_state,
                        ncv, delta, reortho=False):
    """
    do a lanczos simulation (using a tf.while_loop)

    Parameters:
    -------------------------

    ...... fill in ....

    Returns:
    ----------------------------
    (vecs,alpha,beta)
    """
    dtype = initial_state.dtype
    LanczosTridiag = collections.namedtuple(
        'LanczosTridiag',
        ['krylov_vectors', 'UN_krylov_vectors', 'alpha', 'beta'])

    def update_state(old, n, Hxn, xn, alpha, beta):
        return LanczosTridiag(
            krylov_vectors=old.krylov_vectors.write(n + 1, xn),
            UN_krylov_vectors=old.UN_krylov_vectors.write(n + 1, Hxn),
            alpha=old.alpha.write(n, alpha),
            beta=old.beta.write(n + 1, beta))

    def initialize_state(initial):
        un_kv = tf.TensorArray(
            dtype=initial_state.dtype,
            size=ncv + 1,
            name='un_krylov_vectors',
            #infer_shape=False,
            clear_after_read=True)
        beta = tf.linalg.norm(initial)
        initial = tf.math.divide(initial, beta)
        un_kv = un_kv.write(0, initial)
        kv = tf.TensorArray(
            dtype=initial_state.dtype,
            size=ncv + 1,
            name='krylov_vectors',
            #infer_shape=False,            
            clear_after_read=False)
        kv = kv.write(0, tf.zeros(tf.shape(initial), initial.dtype))
        b = tf.TensorArray(
            dtype=initial_state.dtype,
            size=ncv + 1,
            name='beta',
            #infer_shape=False,            
            clear_after_read=False)
        b = b.write(0, beta)
        return LanczosTridiag(
            krylov_vectors=kv,
            UN_krylov_vectors=un_kv,
            alpha=tf.TensorArray(
                dtype=initial_state.dtype,
                size=ncv,
                name='alpha',
                #infer_shape=False,                
                clear_after_read=False),
            beta=b)

    #see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/solvers/python/ops/lanczos.py
    def gram_schmidt_step(j, basis, v):
        """Makes v orthogonal to the j'th vector in basis."""
        #v_shape = v.get_shape()
        basis_vec = basis.read(j)
        v -=  ncon.ncon([tf.reshape(tf.conj(basis_vec), [basis_vec.shape[0] * basis_vec.shape[1] * basis_vec.shape[2]]),
                         tf.reshape(v, [v.shape[0] * v.shape[1] * v.shape[2]])], [[1], [1]])* basis_vec
        #v.set_shape(v_shape)
        return j + 1, basis, v

    def orthogonalize(i, basis, v):
        j = constant_op.constant(0, dtype=dtypes.int32)
        _, _, v = control_flow_ops.while_loop(lambda j, basis, v: j < i,
                                              gram_schmidt_step, [j, basis, v])
        v=tf.math.divide(v,tf.linalg.norm(v))
        return v
    
    def stopping(n, lanstate):
        absbeta = tf.abs(lanstate.beta.read(n))
        return tf.less(tf.cast(tf.abs(delta), absbeta.dtype), absbeta)

    def do_lanczos_step(n, lanstate):
        xn = lanstate.UN_krylov_vectors.read(n)
        beta = tf.linalg.norm(xn)
        xn = tf.math.divide(xn, beta)
        if reortho == True:
            orthogonalize(n-1, lanstate.krylov_vectors, xn)
            
        Hxn = ncon.ncon([L, xn, mpo, R],
                        [[1, -1, 2], [1, 3, 4], [2, 5, -2, 3], [4, -3, 5]])
        #alpha=ncon.ncon([tf.conj(xn),Hxn],[[1,2,3],[1,2,3]])
        # alpha = ncon.ncon([
        #     tf.reshape(tf.conj(xn), [xn.shape[0] * xn.shape[1] * xn.shape[2]]),
        #     tf.reshape(Hxn, [Hxn.shape[0] * Hxn.shape[1] * Hxn.shape[2]])
        # ], [[1], [1]])
        alpha = ncon.ncon([tf.conj(xn),Hxn],[[1,2,3],[1,2,3]])
        Hxn = Hxn - tf.multiply(lanstate.krylov_vectors.read(n),
                                beta) - tf.multiply(xn, alpha)
        return n + 1, update_state(
            old=lanstate, n=n, Hxn=Hxn, xn=xn, alpha=alpha, beta=beta)

    def cond(n, lanstate):
        return tf.cond(
            tf.less(0, n), lambda: tf.cond(
                tf.less(n, ncv), lambda: stopping(n, lanstate), lambda: False),
            lambda: True)

    ls = initialize_state(initial_state)
    n_final, ls_final = tf.while_loop(cond, do_lanczos_step, (0, ls))
    #note that self.vecs[0] is a zeros tensors (used for the first iteration); krylov vectors start at index 1
    return n_final, ls_final.krylov_vectors.stack()[1:,:,:,:], ls_final.alpha.stack(
    ), ls_final.beta.stack()[2:]


def do_lanczos_uncompiled(L, mpo, R, initial_state, ncv, delta, reortho=False):

    """
    do a lanczos simulation (using a tf.while_loop)

    Parameters:
    -------------------------

    ...... fill in ....

    Returns:
    ----------------------------
    (vecs,alpha,beta)
    """
    dtype = initial_state.dtype
    LanczosTridiag = collections.namedtuple(
        'LanczosTridiag',
        ['krylov_vectors', 'UN_krylov_vectors', 'alpha', 'beta'])

    def update_state(old, n, Hxn, xn, alpha, beta):
        return LanczosTridiag(
            krylov_vectors=old.krylov_vectors.write(n + 1, xn),
            UN_krylov_vectors=old.UN_krylov_vectors.write(n + 1, Hxn),
            alpha=old.alpha.write(n, alpha),
            beta=old.beta.write(n + 1, beta))

    def initialize_state(initial):
        un_kv = tf.TensorArray(
            dtype=initial_state.dtype,
            size=ncv + 1,
            name='un_krylov_vectors',
            #infer_shape=False,
            clear_after_read=True)
        beta = tf.linalg.norm(initial)        
        initial = tf.math.divide(initial, beta)
        un_kv = un_kv.write(0, initial)
        kv = tf.TensorArray(
            dtype=initial_state.dtype,
            size=ncv + 1,
            name='krylov_vectors',
            #infer_shape=False,            
            clear_after_read=False)
        kv = kv.write(0, tf.zeros(tf.shape(initial), initial.dtype))
        b = tf.TensorArray(
            dtype=initial_state.dtype,
            size=ncv + 1,
            name='beta',
            #infer_shape=False,            
            clear_after_read=False)
        b = b.write(0, beta)
        return LanczosTridiag(
            krylov_vectors=kv,
            UN_krylov_vectors=un_kv,
            alpha=tf.TensorArray(
                dtype=initial_state.dtype,
                size=ncv,
                name='alpha',
                #infer_shape=False,
                clear_after_read=False),
            beta=b)
    
    #see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/solvers/python/ops/lanczos.py
    def gram_schmidt_step(j, basis, v):
        """Makes v orthogonal to the j'th vector in basis."""
        #v_shape = v.get_shape()
        basis_vec = basis.read(j)
        v -=  ncon.ncon([tf.reshape(tf.conj(basis_vec), [basis_vec.shape[0] * basis_vec.shape[1] * basis_vec.shape[2]]),
                         tf.reshape(v, [v.shape[0] * v.shape[1] * v.shape[2]])], [[1], [1]])* basis_vec
        #v.set_shape(v_shape)
        return j + 1, basis, v

    def orthogonalize(i, basis, v):
        j = constant_op.constant(0, dtype=dtypes.int32)
        _, _, v = control_flow_ops.while_loop(lambda j, basis, v: j < i,
                                              gram_schmidt_step, [j, basis, v])
        v=tf.math.divide(v,tf.linalg.norm(v))
        return v

    def stopping(n, lanstate):
        absbeta = tf.abs(lanstate.beta.read(n))
        return tf.less(tf.cast(tf.abs(delta), absbeta.dtype), absbeta)

    def do_lanczos_step(n, lanstate):
        xn = lanstate.UN_krylov_vectors.read(n)
        beta = tf.linalg.norm(xn)
        xn = tf.math.divide(xn, beta)
        if reortho == True:
            orthogonalize(n-1, lanstate.krylov_vectors, xn)

        Hxn = ncon.ncon([L, xn, mpo, R],
                        [[1, -1, 2], [1, 3, 4], [2, 5, -2, 3], [4, -3, 5]])
        #alpha=ncon.ncon([tf.conj(xn),Hxn],[[1,2,3],[1,2,3]])
        alpha = ncon.ncon([tf.conj(xn),Hxn],[[1,2,3],[1,2,3]])        
        # alpha = ncon.ncon([
        #     tf.reshape(tf.conj(xn), [xn.shape[0] * xn.shape[1] * xn.shape[2]]),
        #     tf.reshape(Hxn, [Hxn.shape[0] * Hxn.shape[1] * Hxn.shape[2]])
        # ], [[1], [1]])
        Hxn = Hxn - tf.multiply(lanstate.krylov_vectors.read(n),
                                beta) - tf.multiply(xn, alpha)
        return n + 1, update_state(
            old=lanstate, n=n, Hxn=Hxn, xn=xn, alpha=alpha, beta=beta)

    def cond(n, lanstate):
        return tf.cond(
            tf.less(0, n), lambda: tf.cond(
                tf.less(n, ncv), lambda: stopping(n, lanstate), lambda: False),
            lambda: True)

    ls = initialize_state(initial_state)
    n_final, ls_final = tf.while_loop(cond, do_lanczos_step, (0, ls))
    #note that self.vecs[0] is a zeros tensors (used for the first iteration); krylov vectors start at index 1
    return n_final, ls_final.krylov_vectors.stack()[1:,:,:,:], ls_final.alpha.stack(
    ), ls_final.beta.stack()[2:]




@tf.contrib.eager.defun
def do_lanczos_simple(L, mpo, R, initial_state, ncv, delta):
    """
    do a lanczos simulation (using a tf.while_loop)

    Parameters:
    -------------------------

    ...... fill in ....

    Returns:
    ----------------------------
    (vecs,alpha,beta)
    """


    
    dtype = initial_state.dtype
    vecs = [tf.math.multiply(initial_state, 0.0)]
    vecs.append(initial_state)
    alphas, betas = [], []
    #tf.contrib.autograph.set_element_type(alphas, dtype)
    #tf.contrib.autograph.set_element_type(betas, dtype)    
    #betas.append(tf.linalg.norm(vecs[0]))    
    for n in range(ncv):
        xn = vecs[n+1]
        beta = tf.linalg.norm(xn)
        betas.append(beta)
        xn = tf.math.divide(xn, beta)
        vecs[n+1]=xn
        Hxn = ncon.ncon([L, xn, mpo, R],
                        [[1, -1, 2], [1, 3, 4], [2, 5, -2, 3], [4, -3, 5]])
        alpha = ncon.ncon([
            tf.reshape(tf.conj(xn), [xn.shape[0] * xn.shape[1] * xn.shape[2]]),
            tf.reshape(Hxn, [Hxn.shape[0] * Hxn.shape[1] * Hxn.shape[2]])
        ], [[1], [1]])
        alphas.append(alpha)
        Hxn = Hxn - tf.multiply(vecs[n],beta) - tf.multiply(xn, alpha)
        vecs.append(Hxn)            
    return ncv, vecs[1:-1], alphas, betas[1:]
    #return ncv, vecs[1:-1],tf.contrib.autograph.stack(alphas),tf.contrib.autograph.stack(betas)


@tf.contrib.eager.defun
def do_lanczos_simple_tensorarray(L, mpo, R, initial_state, ncv, delta):
    """
    do a lanczos simulation (using a tf.while_loop)

    Parameters:
    -------------------------

    ...... fill in ....

    Returns:
    ----------------------------
    (vecs,alpha,beta)
    """


    dtype = initial_state.dtype
    vecs = tf.TensorArray(dtype, element_shape=initial_state.shape, size=ncv+1, clear_after_read=False,)
    vecs = vecs.write(0, tf.zeros(shape = initial_state.shape, dtype=dtype))
    Hxn = initial_state
    alphas = tf.TensorArray(dtype, ncv)
    betas = tf.TensorArray(dtype, ncv)   
    
    for n in range(ncv):
        xn = Hxn#vecs[n+1]
        beta = tf.linalg.norm(xn)
        betas = betas.write(n, beta)
        xn = tf.math.divide(xn, beta)
        vecs = vecs.write(n+1,xn)#[n+1]=xn
        Hxn = ncon.ncon([L, xn, mpo, R],
                        [[1, -1, 2], [1, 3, 4], [2, 5, -2, 3], [4, -3, 5]])
        alpha = ncon.ncon([
            tf.reshape(tf.conj(xn), [xn.shape[0] * xn.shape[1] * xn.shape[2]]),
            tf.reshape(Hxn, [Hxn.shape[0] * Hxn.shape[1] * Hxn.shape[2]])
        ], [[1], [1]])
        alphas = alphas.write(n,alpha)
        Hxn = Hxn - tf.multiply(vecs.read(n),beta) - tf.multiply(xn, alpha)
        #last = Hxn#vecs.append(Hxn)            
    #return vecs[1:-1],tf.contrib.autograph.stack(alphas),tf.contrib.autograph.stack(betas)
    return ncv, vecs.stack()[1:],alphas.stack(),betas.stack()[1:]
    

def tridiag_tensorflow(vecs, alpha, beta):
    Heff=tf.contrib.distributions.tridiag(beta, alpha, tf.conj(beta))
    eta, u = tf.linalg.eigh(Heff)  #could use tridiag
    out=ncon.ncon([vecs,u],[[1,-1,-2,-3],[1,-4]])
    out=out[:,:,:,0]
    out=tf.math.divide(out,tf.linalg.norm(out))
    return eta[0], out


class LanczosEngine:
    """
    This is a general purpose Lanczos-class. It performs a Lanczos tridiagonalization 
    of a hermitian sparse Hamiltonian, defined by the matrix-vector product matvec. 
    """

    def __init__(self, matvec, Ndiag, ncv, delta, deltaEta):
        """
        initialize a Lanczos optimization
        Parameters:
        ---------------------
        matvec:   callable
                  matrix vector multiplication
        Ndiag:    int
                  step at which tridiag Hamiltonian is diagonalized
        ncv:      int
                  number of krylov steps
        delta:    float
                  orthogonality threshold; once the next vector of the iteration is orthogonal to the previous ones 
                  within ```delta``` precision, iteration is terminated
        deltaEta: float
                  desired precision of the energies; once eigenvalues of tridiad Hamiltonian are converged within ```deltaEta```
                  iteration is terminated

        Returns:
        ---------------------
        (eta,states)

        eta: tf.Tensor of shape (numeig)
             the eigenvalues of the tridiag Hamiltonian
        states: list of tf.Tensor 
                the eigenvectors (in tensor format) of the tridiag hamiltonian
        """

        self.Ndiag = Ndiag
        self.ncv = ncv
        self.delta = delta
        self.deltaEta = deltaEta
        self.matvec = matvec
        assert (Ndiag > 0)

    def _simulate(self, initialstate, reortho=False, verbose=False):
        """
        do a lanczos simulation

        Parameters:
        -------------------------
        initialstate: tf.Tensor,
                      the initial state
        reortho:      bool
                      if True, krylov vectors are reorthogonalized at each step (costly)
                      the current implementation is not optimal: there are better ways to do this
        verbose:      bool
        verbosity flag
        """
        self.delta = tf.cast(self.delta, initialstate.dtype)
        self.deltaEta = tf.cast(self.deltaEta, initialstate.dtype)

        dtype = self.matvec(initialstate).dtype
        #initialization:
        xn = copy.deepcopy(initialstate)
        xn /= tf.sqrt(
            ncon.ncon([tf.conj(xn), xn],
                      [range(len(xn.shape)),
                       range(len(xn.shape))]))

        xn_minus_1 = tf.zeros(initialstate.shape, dtype=dtype)
        converged = False
        it = 0
        kn = []
        epsn = []
        self.vecs = []
        first = True
        while converged == False:
            knval = tf.sqrt(
                ncon.ncon([tf.conj(xn), xn],
                          [range(len(xn.shape)),
                           range(len(xn.shape))]))
            if tf.cond(
                    tf.less(tf.abs(knval),
                            tf.abs(self.delta)), lambda: True, lambda: False):
                break
            kn.append(knval)
            xn = xn / kn[-1]
            #store the Lanczos vector for later

            if reortho == True:
                for v in self.vecs:
                    xn -= ncon.ncon([tf.conj(v), xn],
                                    [range(len(v.shape)),
                                     range(len(xn.shape))]) * v
            self.vecs.append(xn)
            Hxn = self.matvec(xn)
            epsn.append(
                ncon.ncon([tf.conj(xn), Hxn],
                          [range(len(xn.shape)),
                           range(len(Hxn.shape))]))
            if ((it % self.Ndiag) == 0) & (len(epsn) >= 1):
                #diagonalize the effective Hamiltonian

                Heff = tf.convert_to_tensor(
                    np.diag(epsn) + np.diag(kn[1:], 1) + np.diag(
                        tf.conj(kn[1:]), -1),
                    dtype=dtype)
                eta, u = tf.linalg.eigh(Heff)  #could use a tridiag solver
                if first == False:
                    if tf.abs(tf.linalg.norm(eta - etaold)) < tf.abs(
                            self.deltaEta):
                        converged = True
                first = False
                etaold = eta[0]
            if it > 0:
                Hxn -= (self.vecs[-1] * epsn[-1])
                Hxn -= (self.vecs[-2] * kn[-1])
            else:
                Hxn -= (self.vecs[-1] * epsn[-1])
            xn = Hxn
            it = it + 1
            if it > self.ncv:
                break
        self.Heff = tf.convert_to_tensor(
            np.diag(epsn) + np.diag(kn[1:], 1) + np.diag(np.conj(kn[1:]), -1),
            dtype=dtype)
        eta, u = tf.linalg.eigh(self.Heff)  #could use tridiag
        states = []
        for n2 in range(min(1, eta.shape[0])):
            state = tf.zeros(initialstate.shape, dtype=initialstate.dtype)
            for n1 in range(len(self.vecs)):
                state += self.vecs[n1] * u[n1, n2]
            states.append(state / tf.sqrt(
                ncon.ncon([tf.conj(state), state],
                          [range(len(state.shape)),
                           range(len(state.shape))])))
        return eta[0], states[0], converged


def compile_lanczos(on=True, simple=False):
    global do_lanczos, tridiag
    tridiag = tridiag_tensorflow                       
    if on:
        if simple:
            do_lanczos = do_lanczos_simple_tensorarray
        else:
            do_lanczos = do_lanczos_compiled
    else:
        do_lanczos = do_lanczos_uncompiled 
        

compile_lanczos(on=True,simple=False)


    
