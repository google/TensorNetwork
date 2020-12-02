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

import functools
from typing import List, Any, Tuple, Callable, Sequence, Text
import collections
import types
import numpy as np
Tensor = Any


def _iterative_classical_gram_schmidt(jax: types.ModuleType) -> Callable:

  JaxPrecisionType = type(jax.lax.Precision.DEFAULT)
  def iterative_classical_gram_schmidt(
      vector: jax.ShapedArray,
      krylov_vectors: jax.ShapedArray,
      precision: JaxPrecisionType,
      iterations: int = 2,
      ) -> jax.ShapedArray:
    """
    Orthogonalize `vector`  to all rows of `krylov_vectors`.

    Args:
      vector: Initial vector.
      krylov_vectors: Matrix of krylov vectors, each row is treated as a
        vector.
      iterations: Number of iterations.

    Returns:
      jax.ShapedArray: The orthogonalized vector.
    """
    i1 = list(range(1, len(krylov_vectors.shape)))
    i2 = list(range(len(vector.shape)))

    vec = vector
    overlaps = 0
    for _ in range(iterations):
      ov = jax.numpy.tensordot(
          krylov_vectors.conj(), vec, (i1, i2), precision=precision)
      vec = vec - jax.numpy.tensordot(
          ov, krylov_vectors, ([0], [0]), precision=precision)
      overlaps = overlaps + ov
    return vec, overlaps

  return iterative_classical_gram_schmidt


def _generate_jitted_eigsh_lanczos(jax: types.ModuleType) -> Callable:
  """
  Helper function to generate jitted lanczos function used
  in JaxBackend.eigsh_lanczos. The function `jax_lanczos`
  returned by this higher-order function has the following
  call signature:
  ```
  eigenvalues, eigenvectors = jax_lanczos(matvec:Callable,
                                     arguments: List[Tensor],
                                     init: Tensor,
                                     ncv: int,
                                     neig: int,
                                     landelta: float,
                                     reortho: bool)
  ```
  `matvec`: A callable implementing the matrix-vector product of a
  linear operator. `arguments`: Arguments to `matvec` additional to
  an input vector. `matvec` will be called as `matvec(init, *args)`.
  `init`: An initial input vector to `matvec`.
  `ncv`: Number of krylov iterations (i.e. dimension of the Krylov space).
  `neig`: Number of eigenvalue-eigenvector pairs to be computed.
  `landelta`: Convergence parameter: if the norm of the current Lanczos vector

  `reortho`: If `True`, reorthogonalize all krylov vectors at each step.
     This should be used if `neig>1`.

  Args:
    jax: The `jax` module.
  Returns:
    Callable: A jitted function that does a lanczos iteration.

  """
  JaxPrecisionType = type(jax.lax.Precision.DEFAULT)

  @functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
  def jax_lanczos(matvec: Callable, arguments: List, init: jax.ShapedArray,
                  ncv: int, neig: int, landelta: float, reortho: bool,
                  precision: JaxPrecisionType) -> Tuple[jax.ShapedArray, List]:
    """
    Lanczos iteration for symmeric eigenvalue problems. If reortho = False,
    the Krylov basis is constructed without explicit re-orthogonalization. 
    In infinite precision, all Krylov vectors would be orthogonal. Due to 
    finite precision arithmetic, orthogonality is usually quickly lost. 
    For reortho=True, the Krylov basis is explicitly reorthogonalized.

    Args:
      matvec: A callable implementing the matrix-vector product of a
        linear operator.
      arguments: Arguments to `matvec` additional to an input vector.
        `matvec` will be called as `matvec(init, *args)`.
      init: An initial input vector to `matvec`.
      ncv: Number of krylov iterations (i.e. dimension of the Krylov space).
      neig: Number of eigenvalue-eigenvector pairs to be computed.
      landelta: Convergence parameter: if the norm of the current Lanczos vector
        falls below `landelta`, iteration is stopped.
      reortho: If `True`, reorthogonalize all krylov vectors at each step.
        This should be used if `neig>1`.
      precision: jax.lax.Precision type used in jax.numpy.vdot

    Returns:
      jax.ShapedArray: Eigenvalues
      List: Eigenvectors
      int: Number of iterations
    """
    shape = init.shape
    dtype = init.dtype
    iterative_classical_gram_schmidt = _iterative_classical_gram_schmidt(jax)
    mask_slice = (slice(ncv + 2), ) + (None,) * len(shape)
    def scalar_product(a, b):
      i1 = list(range(len(a.shape)))
      i2 = list(range(len(b.shape)))
      return jax.numpy.tensordot(a.conj(), b, (i1, i2), precision=precision)

    def norm(a):
      return jax.numpy.sqrt(scalar_product(a, a))

    def body_lanczos(vals):
      krylov_vectors, alphas, betas, i = vals
      previous_vector = krylov_vectors[i]
      def body_while(vals):
        pv, kv, _ = vals
        pv = iterative_classical_gram_schmidt(
            pv, (i > jax.numpy.arange(ncv + 2))[mask_slice] * kv, precision)[0]
        return [pv, kv, False]

      def cond_while(vals):
        return vals[2]

      previous_vector, krylov_vectors, _ = jax.lax.while_loop(
          cond_while, body_while,
          [previous_vector, krylov_vectors, reortho])

      beta = norm(previous_vector)
      normalized_vector = previous_vector / beta
      Av = matvec(normalized_vector, *arguments)
      alpha = scalar_product(normalized_vector, Av)
      alphas = alphas.at[i - 1].set(alpha)
      betas = betas.at[i].set(beta)

      def while_next(vals):
        Av, _ = vals
        res = Av - normalized_vector * alpha -   krylov_vectors[i - 1] * beta
        return [res, False]

      def cond_next(vals):
        return vals[1]

      next_vector, _ = jax.lax.while_loop(
          cond_next, while_next,
          [Av, jax.numpy.logical_not(reortho)])
      next_vector = jax.numpy.reshape(next_vector, shape)

      krylov_vectors = krylov_vectors.at[i].set(normalized_vector)
      krylov_vectors = krylov_vectors.at[i + 1].set(next_vector)

      return [krylov_vectors, alphas, betas, i + 1]

    def cond_fun(vals):
      betas, i = vals[-2], vals[-1]
      norm = betas[i - 1]
      return jax.lax.cond(i <= ncv, lambda x: x[0] > x[1], lambda x: False,
                          [norm, landelta])

    # note: ncv + 2 because the first vector is all zeros, and the
    # last is the unnormalized residual.
    krylov_vecs = jax.numpy.zeros((ncv + 2,) + shape, dtype=dtype)
    # NOTE (mganahl): initial vector is normalized inside the loop
    krylov_vecs = krylov_vecs.at[1].set(init)

    # betas are the upper and lower diagonal elements
    # of the projected linear operator
    # the first two beta-values can be discarded
    # set betas[0] to 1.0 for initialization of loop
    # betas[2] is set to the norm of the initial vector.
    betas = jax.numpy.zeros(ncv + 1, dtype=dtype)
    betas = betas.at[0].set(1.0)
    # diagonal elements of the projected linear operator
    alphas = jax.numpy.zeros(ncv, dtype=dtype)
    initvals = [krylov_vecs, alphas, betas, 1]
    krylov_vecs, alphas, betas, numits = jax.lax.while_loop(
        cond_fun, body_lanczos, initvals)
    # FIXME (mganahl): if the while_loop stopps early at iteration i, alphas
    # and betas are 0.0 at positions n >= i - 1. eigh will then wrongly give
    # degenerate eigenvalues 0.0. JAX does currently not support
    # dynamic slicing with variable slice sizes, so these beta values
    # can't be truncated. Thus, if numeig >= i - 1, jitted_lanczos returns
    # a set of spurious eigen vectors and eigen values.
    # If algebraically small EVs are desired, one can initialize `alphas` with
    # large positive values, thus pushing the spurious eigenvalues further
    # away from the desired ones (similar for algebraically large EVs)

    #FIXME: replace with eigh_banded once JAX supports it
    A_tridiag = jax.numpy.diag(alphas) + jax.numpy.diag(
        betas[2:], 1) + jax.numpy.diag(jax.numpy.conj(betas[2:]), -1)
    eigvals, U = jax.numpy.linalg.eigh(A_tridiag)
    eigvals = eigvals.astype(dtype)

    # expand eigenvectors in krylov basis
    def body_vector(i, vals):
      krv, unitary, vectors = vals
      dim = unitary.shape[1]
      n, m = jax.numpy.divmod(i, dim)
      vectors = jax.ops.index_add(vectors, jax.ops.index[n, :],
                                  krv[m + 1] * unitary[m, n])
      return [krv, unitary, vectors]

    _vectors = jax.numpy.zeros((neig,) + shape, dtype=dtype)
    _, _, vectors = jax.lax.fori_loop(0, neig * (krylov_vecs.shape[0] - 1),
                                      body_vector,
                                      [krylov_vecs, U, _vectors])

    return jax.numpy.array(eigvals[0:neig]), [
        vectors[n] / norm(vectors[n]) for n in range(neig)
    ], numits

  return jax_lanczos


def _generate_lanczos_factorization(jax: types.ModuleType) -> Callable:
  """
  Helper function to generate a jitteed function that 
  computes a lanczos factoriazation of a linear operator.
  Returns:
    Callable: A jitted function that does a lanczos factorization.

  """
  JaxPrecisionType = type(jax.lax.Precision.DEFAULT)

  @functools.partial(jax.jit, static_argnums=(6, 7, 8, 9))
  def _lanczos_fact(
      matvec: Callable, args: List, v0: jax.ShapedArray,
      Vm: jax.ShapedArray, alphas: jax.ShapedArray, betas: jax.ShapedArray,
      start: int, num_krylov_vecs: int, tol: float, precision: JaxPrecisionType
  ):
    """
    Compute an m-step lanczos factorization of `matvec`, with
    m <=`num_krylov_vecs`. The factorization will
    do at most `num_krylov_vecs` steps, and terminate early 
    if an invariat subspace is encountered. The returned arrays
    `alphas`, `betas` and `Vm` will satisfy the Lanczos recurrence relation
    ```
    matrix @ Vm - Vm @ Hm - fm * em = 0
    ```
    with `matrix` the matrix representation of `matvec`, 
    `Hm = jnp.diag(alphas) + jnp.diag(betas, -1) + jnp.diag(betas.conj(), 1)`
    `fm=residual * norm`, and `em` a cartesian basis vector of shape 
    `(1, kv.shape[1])` with `em[0, -1] == 1` and 0 elsewhere.

    Note that the caller is responsible for dtype consistency between
    the inputs, i.e. dtypes between all input arrays have to match.

    Args:
      matvec: The matrix vector product.
      args: List of arguments to `matvec`.
      v0: Initial state to `matvec`.
      Vm: An array for storing the krylov vectors. The individual
        vectors are stored as columns.
        The shape of `krylov_vecs` has to be
        (num_krylov_vecs + 1, np.ravel(v0).shape[0]).
      alphas: An array for storing the diagonal elements of the reduced
        operator.
      betas: An array for storing the lower diagonal elements of the 
        reduced operator.
      start: Integer denoting the start position where the first
        produced krylov_vector should be inserted into `Vm`
      num_krylov_vecs: Number of krylov iterations, should be identical to
        `Vm.shape[0] + 1`
      tol: Convergence parameter. Iteration is terminated if the norm of a
        krylov-vector falls below `tol`.

    Returns:
      jax.ShapedArray: An array of shape 
        `(num_krylov_vecs, np.prod(initial_state.shape))` of krylov vectors.
      jax.ShapedArray: The diagonal elements of the tridiagonal reduced
        operator ("alphas")
      jax.ShapedArray: The lower-diagonal elements of the tridiagonal reduced
        operator ("betas")
      jax.ShapedArray: The unnormalized residual of the Lanczos process.
      float: The norm of the residual.
      int: The number of performed iterations.
      bool: if `True`: iteration hit an invariant subspace.
            if `False`: iteration terminated without encountering
            an invariant subspace.
    """

    shape = v0.shape
    iterative_classical_gram_schmidt = _iterative_classical_gram_schmidt(jax)
    Z = jax.numpy.linalg.norm(v0)
    #only normalize if norm > tol, else return zero vector
    v = jax.lax.cond(Z > tol, lambda x: v0 / Z, lambda x: v0 * 0.0, None)
    Vm = Vm.at[start, :].set(jax.numpy.ravel(v))
    betas = jax.lax.cond(
        start > 0,
        lambda x: betas.at[start - 1].set(Z),
        lambda x: betas, start)
    # body of the arnoldi iteration
    def body(vals):
      Vm, alphas, betas, previous_vector, _, i = vals
      Av = matvec(previous_vector, *args)
      Av, overlaps = iterative_classical_gram_schmidt(
          Av.ravel(),
          (i >= jax.numpy.arange(Vm.shape[0]))[:, None] * Vm, precision)
      alphas = alphas.at[i].set(overlaps[i])
      norm = jax.numpy.linalg.norm(Av)
      Av = jax.numpy.reshape(Av, shape)
      # only normalize if norm is larger than threshold,
      # otherwise return zero vector
      Av = jax.lax.cond(norm > tol, lambda x: Av/norm, lambda x: Av * 0.0, None)
      Vm, betas = jax.lax.cond(
          i < num_krylov_vecs - 1,
          lambda x: (Vm.at[i + 1, :].set(Av.ravel()), betas.at[i].set(norm)),
          lambda x: (Vm, betas),
          None)

      return [Vm, alphas, betas, Av, norm, i + 1]

    def cond_fun(vals):
      # Continue loop while iteration < num_krylov_vecs and norm > tol
      norm, iteration = vals[4], vals[5]
      counter_done = (iteration >= num_krylov_vecs)
      norm_not_too_small = norm > tol
      continue_iteration = jax.lax.cond(counter_done, lambda x: False,
                                        lambda x: norm_not_too_small, None)
      return continue_iteration
    initial_values = [Vm, alphas, betas, v, Z, start]
    final_values = jax.lax.while_loop(cond_fun, body, initial_values)
    Vm, alphas, betas, residual, norm, it = final_values
    return Vm, alphas, betas, residual, norm, it, norm < tol

  return _lanczos_fact


def _generate_arnoldi_factorization(jax: types.ModuleType) -> Callable:
  """
  Helper function to create a jitted arnoldi factorization.
  The function returns a function `_arnoldi_fact` which
  performs an m-step arnoldi factorization.

  `_arnoldi_fact` computes an m-step arnoldi factorization
  of an input callable `matvec`, with m = min(`it`,`num_krylov_vecs`).
  `_arnoldi_fact` will do at most `num_krylov_vecs` steps.
  `_arnoldi_fact` returns arrays `kv` and `H` which satisfy
  the Arnoldi recurrence relation
  ```
  matrix @ Vm - Vm @ Hm - fm * em = 0
  ```
  with `matrix` the matrix representation of `matvec` and
  `Vm =  jax.numpy.transpose(kv[:it, :])`,
  `Hm = H[:it, :it]`, `fm = np.expand_dims(kv[it, :] * H[it, it - 1]`,1)
  and `em` a kartesian basis vector of shape `(1, kv.shape[1])`
  with `em[0, -1] == 1` and 0 elsewhere.

  Note that the caller is responsible for dtype consistency between
  the inputs, i.e. dtypes between all input arrays have to match.

  Args:
    matvec: The matrix vector product. This function has to be wrapped into
      `jax.tree_util.Partial`. `matvec` will be called as `matvec(x, *args)`
      for an input vector `x`.
    args: List of arguments to `matvec`.
    v0: Initial state to `matvec`.
    Vm: An array for storing the krylov vectors. The individual
      vectors are stored as columns. The shape of `krylov_vecs` has to be
      (num_krylov_vecs + 1, np.ravel(v0).shape[0]).
    H: Matrix of overlaps. The shape has to be
      (num_krylov_vecs + 1,num_krylov_vecs + 1).
    start: Integer denoting the start position where the first
      produced krylov_vector should be inserted into `Vm`
    num_krylov_vecs: Number of krylov iterations, should be identical to
      `Vm.shape[0] + 1`
    tol: Convergence parameter. Iteration is terminated if the norm of a
      krylov-vector falls below `tol`.

  Returns:
    kv: An array of krylov vectors
    H: A matrix of overlaps
    it: The number of performed iterations.
    converged: Whether convergence was achieved.

  """
  JaxPrecisionType = type(jax.lax.Precision.DEFAULT)
  iterative_classical_gram_schmidt = _iterative_classical_gram_schmidt(jax)

  @functools.partial(jax.jit, static_argnums=(5, 6, 7, 8))
  def _arnoldi_fact(
      matvec: Callable, args: List, v0: jax.ShapedArray,
      Vm: jax.ShapedArray, H: jax.ShapedArray, start: int,
      num_krylov_vecs: int, tol: float, precision: JaxPrecisionType
  ) -> Tuple[jax.ShapedArray, jax.ShapedArray, jax.ShapedArray, float, int,
             bool]:
    """
    Compute an m-step arnoldi factorization of `matvec`, with
    m = min(`it`,`num_krylov_vecs`). The factorization will
    do at most `num_krylov_vecs` steps. The returned arrays
    `kv` and `H` will satisfy the Arnoldi recurrence relation
    ```
    matrix @ Vm - Vm @ Hm - fm * em = 0
    ```
    with `matrix` the matrix representation of `matvec` and
    `Vm =  jax.numpy.transpose(kv[:it, :])`,
    `Hm = H[:it, :it]`, `fm = np.expand_dims(kv[it, :] * H[it, it - 1]`,1)
    and `em` a cartesian basis vector of shape `(1, kv.shape[1])`
    with `em[0, -1] == 1` and 0 elsewhere.

    Note that the caller is responsible for dtype consistency between
    the inputs, i.e. dtypes between all input arrays have to match.

    Args:
      matvec: The matrix vector product.
      args: List of arguments to `matvec`.
      v0: Initial state to `matvec`.
      Vm: An array for storing the krylov vectors. The individual
        vectors are stored as columns.
        The shape of `krylov_vecs` has to be
        (num_krylov_vecs + 1, np.ravel(v0).shape[0]).
      H: Matrix of overlaps. The shape has to be
        (num_krylov_vecs + 1,num_krylov_vecs + 1).
      start: Integer denoting the start position where the first
        produced krylov_vector should be inserted into `Vm`
      num_krylov_vecs: Number of krylov iterations, should be identical to
        `Vm.shape[0] + 1`
      tol: Convergence parameter. Iteration is terminated if the norm of a
        krylov-vector falls below `tol`.
    Returns:
      jax.ShapedArray: An array of shape 
        `(num_krylov_vecs, np.prod(initial_state.shape))` of krylov vectors.
      jax.ShapedArray: Upper Hessenberg matrix of shape 
        `(num_krylov_vecs, num_krylov_vecs`) of the Arnoldi processs.
      jax.ShapedArray: The unnormalized residual of the Arnoldi process.
      int: The norm of the residual.
      int: The number of performed iterations.
      bool: if `True`: iteration hit an invariant subspace.
            if `False`: iteration terminated without encountering
            an invariant subspace.
    """

    # Note (mganahl): currently unused, but is very convenient to have
    # for further development and tests (it's usually more accurate than
    # classical gs)
    # Call signature:
    #```python
    # initial_vals = [Av.ravel(), Vm, i, H]
    # Av, Vm, _, H = jax.lax.fori_loop(
    #     0, i + 1, modified_gram_schmidt_step_arnoldi, initial_vals)
    #```
    def modified_gram_schmidt_step_arnoldi(j, vals): #pylint: disable=unused-variable
      """
      Single step of a modified gram-schmidt orthogonalization.
      Substantially more accurate than classical gram schmidt
      Args:
        j: Integer value denoting the vector to be orthogonalized.
        vals: A list of variables:
          `vector`: The current vector to be orthogonalized
          to all previous ones
          `Vm`: jax.array of collected krylov vectors
          `n`: integer denoting the column-position of the overlap
            <`krylov_vector`|`vector`> within `H`.
      Returns:
        updated vals.
  
      """
      vector, krylov_vectors, n, H = vals
      v = krylov_vectors[j, :]
      h = jax.numpy.vdot(v, vector, precision=precision)
      H = H.at[j, n].set(h)
      vector = vector - h * v
      return [vector, krylov_vectors, n, H]

    shape = v0.shape
    Z = jax.numpy.linalg.norm(v0)
    #only normalize if norm > tol, else return zero vector
    v = jax.lax.cond(Z > tol, lambda x: v0 / Z, lambda x: v0 * 0.0, None)
    Vm = Vm.at[start, :].set(jax.numpy.ravel(v))
    H = jax.lax.cond(
        start > 0,
        lambda x: H.at[x, x - 1].set(Z),
        lambda x: H, start)
    # body of the arnoldi iteration
    def body(vals):
      Vm, H, previous_vector, _, i = vals
      Av = matvec(previous_vector, *args)

      Av, overlaps = iterative_classical_gram_schmidt(
          Av.ravel(),
          (i >= jax.numpy.arange(Vm.shape[0]))[:, None] *
          Vm, precision)
      H = H.at[:, i].set(overlaps)
      norm = jax.numpy.linalg.norm(Av)
      Av = jax.numpy.reshape(Av, shape)

      # only normalize if norm is larger than threshold,
      # otherwise return zero vector
      Av = jax.lax.cond(norm > tol, lambda x: Av/norm, lambda x: Av * 0.0, None)
      Vm, H = jax.lax.cond(
          i < num_krylov_vecs - 1,
          lambda x: (Vm.at[i + 1, :].set(Av.ravel()), H.at[i + 1, i].set(norm)), #pylint: disable=line-too-long
          lambda x: (x[0], x[1]),
          (Vm, H, Av, i, norm))

      return [Vm, H, Av, norm, i + 1]

    def cond_fun(vals):
      # Continue loop while iteration < num_krylov_vecs and norm > tol
      norm, iteration = vals[3], vals[4]
      counter_done = (iteration >= num_krylov_vecs)
      norm_not_too_small = norm > tol
      continue_iteration = jax.lax.cond(counter_done, lambda x: False,
                                        lambda x: norm_not_too_small, None)
      return continue_iteration

    initial_values = [Vm, H, v, Z, start]
    final_values = jax.lax.while_loop(cond_fun, body, initial_values)
    Vm, H, residual, norm, it = final_values
    return Vm, H, residual, norm, it, norm < tol

  return _arnoldi_fact

# ######################################################
# #######  NEW SORTING FUCTIONS INSERTED HERE  #########
# ######################################################
def _LR_sort(jax):
  @functools.partial(jax.jit, static_argnums=(0,))
  def sorter(
      p: int,
      evals: jax.ShapedArray) -> Tuple[jax.ShapedArray, jax.ShapedArray]:
    inds = jax.numpy.argsort(jax.numpy.real(evals), kind='stable')[::-1]
    shifts = evals[inds][-p:]
    return shifts, inds
  return sorter

def _SA_sort(jax):
  @functools.partial(jax.jit, static_argnums=(0,))
  def sorter(
      p: int,
      evals: jax.ShapedArray) -> Tuple[jax.ShapedArray, jax.ShapedArray]:
    inds = jax.numpy.argsort(evals, kind='stable')
    shifts = evals[inds][-p:]
    return shifts, inds
  return sorter

def _LA_sort(jax):
  @functools.partial(jax.jit, static_argnums=(0,))
  def sorter(
      p: int,
      evals: jax.ShapedArray) -> Tuple[jax.ShapedArray, jax.ShapedArray]:
    inds = jax.numpy.argsort(evals, kind='stable')[::-1]
    shifts = evals[inds][-p:]
    return shifts, inds
  return sorter

def _LM_sort(jax):
  @functools.partial(jax.jit, static_argnums=(0,))
  def sorter(
      p: int,
      evals: jax.ShapedArray) -> Tuple[jax.ShapedArray, jax.ShapedArray]:
    inds = jax.numpy.argsort(jax.numpy.abs(evals), kind='stable')[::-1]
    shifts = evals[inds][-p:]
    return shifts, inds
  return sorter

# ####################################################
# ####################################################

def _shifted_QR(jax):
  @functools.partial(jax.jit, static_argnums=(4,))
  def shifted_QR(
      Vm: jax.ShapedArray, Hm: jax.ShapedArray, fm: jax.ShapedArray,
      shifts: jax.ShapedArray,
      numeig: int) -> Tuple[jax.ShapedArray, jax.ShapedArray, jax.ShapedArray]:
    # compress arnoldi factorization
    q = jax.numpy.zeros(Hm.shape[0], dtype=Hm.dtype)
    q = q.at[-1].set(1.0)

    def body(i, vals):
      Vm, Hm, q = vals
      shift = shifts[i] * jax.numpy.eye(Hm.shape[0], dtype=Hm.dtype)
      Qj, R = jax.numpy.linalg.qr(Hm - shift)
      Hm = R @ Qj + shift
      Vm = Qj.T @ Vm
      q = q @ Qj
      return Vm, Hm, q

    Vm, Hm, q = jax.lax.fori_loop(0, shifts.shape[0], body,
                                  (Vm, Hm, q))
    fk = Vm[numeig, :] * Hm[numeig, numeig - 1] + fm * q[numeig - 1]
    return Vm, Hm, fk
  return shifted_QR

def _get_vectors(jax):
  @functools.partial(jax.jit, static_argnums=(3,))
  def get_vectors(Vm: jax.ShapedArray, unitary: jax.ShapedArray,
                  inds: jax.ShapedArray, numeig: int) -> jax.ShapedArray:

    def body_vector(i, states):
      dim = unitary.shape[1]
      n, m = jax.numpy.divmod(i, dim)
      states = jax.ops.index_add(states, jax.ops.index[n, :],
                                 Vm[m, :] * unitary[m, inds[n]])
      return states

    state_vectors = jax.numpy.zeros([numeig, Vm.shape[1]], dtype=Vm.dtype)
    state_vectors = jax.lax.fori_loop(0, numeig * Vm.shape[0], body_vector,
                                      state_vectors)
    state_norms = jax.numpy.linalg.norm(state_vectors, axis=1)
    state_vectors = state_vectors / state_norms[:, None]
    return state_vectors

  return get_vectors

def _check_eigvals_convergence_eigh(jax):
  @functools.partial(jax.jit, static_argnums=(3,))
  def check_eigvals_convergence(beta_m: float, Hm: jax.ShapedArray,
                                Hm_norm: float,
                                tol: float) -> bool:
    eigvals, eigvecs = jax.numpy.linalg.eigh(Hm)
    # TODO (mganahl) confirm that this is a valid matrix norm)
    thresh = jax.numpy.maximum(
        jax.numpy.finfo(eigvals.dtype).eps * Hm_norm,
        jax.numpy.abs(eigvals) * tol)
    vals = jax.numpy.abs(eigvecs[-1, :])
    return jax.numpy.all(beta_m * vals < thresh)

  return check_eigvals_convergence

def _check_eigvals_convergence_eig(jax):
  @functools.partial(jax.jit, static_argnums=(2, 3))
  def check_eigvals_convergence(beta_m: float, Hm: jax.ShapedArray,
                                tol: float, numeig: int) -> bool:
    eigvals, eigvecs = jax.numpy.linalg.eig(Hm)
    # TODO (mganahl) confirm that this is a valid matrix norm)
    Hm_norm = jax.numpy.linalg.norm(Hm)
    thresh = jax.numpy.maximum(
        jax.numpy.finfo(eigvals.dtype).eps * Hm_norm,
        jax.numpy.abs(eigvals[:numeig]) * tol)
    vals = jax.numpy.abs(eigvecs[numeig - 1, :numeig])
    return jax.numpy.all(beta_m * vals < thresh)

  return check_eigvals_convergence

def _implicitly_restarted_arnoldi(jax: types.ModuleType) -> Callable:
  """
  Helper function to generate a jitted function to do an
  implicitly restarted arnoldi factorization of `matvec`. The
  returned routine finds the lowest `numeig`
  eigenvector-eigenvalue pairs of `matvec`
  by alternating between compression and re-expansion of an initial
  `num_krylov_vecs`-step Arnoldi factorization.

  Note: The caller has to ensure that the dtype of the return value
  of `matvec` matches the dtype of the initial state. Otherwise jax
  will raise a TypeError.

  The function signature of the returned function is
    Args:
      matvec: A callable representing the linear operator.
      args: Arguments to `matvec`.  `matvec` is called with
        `matvec(x, *args)` with `x` the input array on which
        `matvec` should act.
      initial_state: An starting vector for the iteration.
      num_krylov_vecs: Number of krylov vectors of the arnoldi factorization.
        numeig: The number of desired eigenvector-eigenvalue pairs.
      which: Which eigenvalues to target. Currently supported: `which = 'LR'`.
      tol: Convergence flag. If the norm of a krylov vector drops below `tol`
        the iteration is terminated.
      maxiter: Maximum number of (outer) iteration steps.
    Returns:
      eta, U: Two lists containing eigenvalues and eigenvectors.

  Args:
    jax: The jax module.
  Returns:
    Callable: A function performing an implicitly restarted
      Arnoldi factorization
  """
  JaxPrecisionType = type(jax.lax.Precision.DEFAULT)

  arnoldi_fact = _generate_arnoldi_factorization(jax)


  @functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
  def implicitly_restarted_arnoldi_method(
      matvec: Callable, args: List, initial_state: jax.ShapedArray,
      num_krylov_vecs: int, numeig: int, which: Text, tol: float, maxiter: int,
      precision: JaxPrecisionType
  ) -> Tuple[jax.ShapedArray, List[jax.ShapedArray], int]:
    """
    Implicitly restarted arnoldi factorization of `matvec`. The routine
    finds the lowest `numeig` eigenvector-eigenvalue pairs of `matvec`
    by alternating between compression and re-expansion of an initial
    `num_krylov_vecs`-step Arnoldi factorization.

    Note: The caller has to ensure that the dtype of the return value
    of `matvec` matches the dtype of the initial state. Otherwise jax
    will raise a TypeError.

    NOTE: Under certain circumstances, the routine can return spurious
    eigenvalues 0.0: if the Arnoldi iteration terminated early
    (after numits < num_krylov_vecs iterations)
    and numeig > numits, then spurious 0.0 eigenvalues will be returned.

    Args:
      matvec: A callable representing the linear operator.
      args: Arguments to `matvec`.  `matvec` is called with
        `matvec(x, *args)` with `x` the input array on which
        `matvec` should act.
      initial_state: An starting vector for the iteration.
      num_krylov_vecs: Number of krylov vectors of the arnoldi factorization.
        numeig: The number of desired eigenvector-eigenvalue pairs.
      which: Which eigenvalues to target.
        Currently supported: `which = 'LR'` (largest real part).
      tol: Convergence flag. If the norm of a krylov vector drops below `tol`
        the iteration is terminated.
      maxiter: Maximum number of (outer) iteration steps.
      precision: jax.lax.Precision used within lax operations.

    Returns:
      jax.ShapedArray: Eigenvalues
      List: Eigenvectors
      int: Number of inner krylov iterations of the last arnoldi
        factorization.
    """
    shape = initial_state.shape
    dtype = initial_state.dtype

    dim = np.prod(shape).astype(np.int32)
    num_expand = num_krylov_vecs - numeig
    if num_krylov_vecs <= numeig < dim:
      raise ValueError(f"num_krylov_vecs must be between numeig <"
                       f" num_krylov_vecs <= dim = {dim},"
                       f" num_krylov_vecs = {num_krylov_vecs}")
    if numeig > dim:
      raise ValueError(f"number of requested eigenvalues numeig = {numeig} "
                       f"is larger than the dimension of the operator "
                       f"dim = {dim}")

    # initialize arrays
    Vm = jax.numpy.zeros(
        (num_krylov_vecs, jax.numpy.ravel(initial_state).shape[0]), dtype=dtype)
    Hm = jax.numpy.zeros((num_krylov_vecs, num_krylov_vecs), dtype=dtype)
    # perform initial arnoldi factorization
    Vm, Hm, residual, norm, numits, ar_converged = arnoldi_fact(
        matvec, args, initial_state, Vm, Hm, 0, num_krylov_vecs, tol, precision)
    fm = residual.ravel() * norm

    # generate needed functions
    shifted_QR = _shifted_QR(jax)
    check_eigvals_convergence = _check_eigvals_convergence_eig(jax)
    get_vectors = _get_vectors(jax)

    # sort_fun returns `num_expand` least relevant eigenvalues
    # (those to be projected out)
    if which == 'LR':
      sort_fun = jax.tree_util.Partial(_LR_sort(jax), num_expand)
    elif which == 'LM':
      sort_fun = jax.tree_util.Partial(_LM_sort(jax), num_expand)
    else:
      raise ValueError(f"which = {which} not implemented")

    it = 1  # we already did one arnoldi factorization
    if maxiter > 1:
      # cast arrays to correct complex dtype
      if Vm.dtype == np.float64:
        dtype = np.complex128
      elif Vm.dtype == np.float32:
        dtype = np.complex64
      elif Vm.dtype == np.complex128:
        dtype = Vm.dtype
      elif Vm.dtype == np.complex64:
        dtype = Vm.dtype
      else:
        raise TypeError(f'dtype {Vm.dtype} not supported')

      Vm = Vm.astype(dtype)
      Hm = Hm.astype(dtype)
      fm = fm.astype(dtype)

    def outer_loop(carry):
      Hm, Vm, fm, it, numits, ar_converged, _, _, = carry
      evals, _ = jax.numpy.linalg.eig(Hm)
      shifts, _ = sort_fun(evals)
      # perform shifted QR iterations to compress arnoldi factorization
      # Note that ||fk|| typically decreases as one iterates the outer loop
      # indicating that iram converges.
      # ||fk|| = \beta_m in reference above
      Vk, Hk, fk = shifted_QR(Vm, Hm, fm, shifts, numeig)
      # reset matrices
      beta_k = jax.numpy.linalg.norm(fk)
      converged = check_eigvals_convergence(beta_k, Hk, tol, numeig)
      Vk = Vk.at[numeig:, :].set(0.0)
      Hk = Hk.at[numeig:, :].set(0.0)
      Hk = Hk.at[:, numeig:].set(0.0)
      def do_arnoldi(vals):
        Vk, Hk, fk, _, _, _, _ = vals
        # restart
        Vm, Hm, residual, norm, numits, ar_converged = arnoldi_fact(
            matvec, args, jax.numpy.reshape(fk, shape), Vk, Hk, numeig,
            num_krylov_vecs, tol, precision)
        fm = residual.ravel() * norm
        return [Vm, Hm, fm, norm, numits, ar_converged, False]

      def cond_arnoldi(vals):
        return vals[6]

      res = jax.lax.while_loop(cond_arnoldi, do_arnoldi, [
          Vk, Hk, fk,
          jax.numpy.linalg.norm(fk), numeig, False,
          jax.numpy.logical_not(converged)
      ])

      Vm, Hm, fm, norm, numits, ar_converged = res[0:6]
      out_vars = [
          Hm, Vm, fm, it + 1, numits, ar_converged, converged, norm
      ]
      return out_vars

    def cond_fun(carry):
      it, ar_converged, converged = carry[3], carry[5], carry[
          6]
      return jax.lax.cond(
          it < maxiter, lambda x: x, lambda x: False,
          jax.numpy.logical_not(jax.numpy.logical_or(converged, ar_converged)))

    converged = False
    carry = [Hm, Vm, fm, it, numits, ar_converged, converged, norm]
    res = jax.lax.while_loop(cond_fun, outer_loop, carry)
    Hm, Vm = res[0], res[1]
    numits, converged = res[4], res[6]
    # if `ar_converged` then `norm`is below convergence threshold
    # set it to 0.0 in this case to prevent `jnp.linalg.eig` from finding a
    # spurious eigenvalue of order `norm`.
    Hm = Hm.at[numits, numits - 1].set(
        jax.lax.cond(converged, lambda x: Hm.dtype.type(0.0), lambda x: x,
                     Hm[numits, numits - 1]))

    # if the Arnoldi-factorization stopped early (after `numit` iterations)
    # before exhausting the allowed size of the Krylov subspace,
    # (i.e. `numit` < 'num_krylov_vecs'), set elements
    # at positions m, n with m, n >= `numit` to 0.0.

    # FIXME (mganahl): under certain circumstances, the routine can still
    # return spurious 0 eigenvalues: if arnoldi terminated early
    # (after numits < num_krylov_vecs iterations)
    # and numeig > numits, then spurious 0.0 eigenvalues will be returned

    Hm = (numits > jax.numpy.arange(num_krylov_vecs))[:, None] * Hm * (
        numits > jax.numpy.arange(num_krylov_vecs))[None, :]
    eigvals, U = jax.numpy.linalg.eig(Hm)
    inds = sort_fun(eigvals)[1][:numeig]
    vectors = get_vectors(Vm, U, inds, numeig)
    return eigvals[inds], [
        jax.numpy.reshape(vectors[n, :], shape)
        for n in range(numeig)
    ], numits

  return implicitly_restarted_arnoldi_method


def _implicitly_restarted_lanczos(jax: types.ModuleType) -> Callable:
  """
  Helper function to generate a jitted function to do an
  implicitly restarted lanczos factorization of `matvec`. The
  returned routine finds the lowest `numeig`
  eigenvector-eigenvalue pairs of `matvec`
  by alternating between compression and re-expansion of an initial
  `num_krylov_vecs`-step Lanczos factorization.

  Note: The caller has to ensure that the dtype of the return value
  of `matvec` matches the dtype of the initial state. Otherwise jax
  will raise a TypeError.

  The function signature of the returned function is
    Args:
      matvec: A callable representing the linear operator.
      args: Arguments to `matvec`.  `matvec` is called with
        `matvec(x, *args)` with `x` the input array on which
        `matvec` should act.
      initial_state: An starting vector for the iteration.
      num_krylov_vecs: Number of krylov vectors of the lanczos factorization.
        numeig: The number of desired eigenvector-eigenvalue pairs.
      which: Which eigenvalues to target. Currently supported: `which = 'LR'`
        or `which = 'SR'`.
      tol: Convergence flag. If the norm of a krylov vector drops below `tol`
        the iteration is terminated.
      maxiter: Maximum number of (outer) iteration steps.
    Returns:
      eta, U: Two lists containing eigenvalues and eigenvectors.

  Args:
    jax: The jax module.
  Returns:
    Callable: A function performing an implicitly restarted
      Lanczos factorization
  """
  JaxPrecisionType = type(jax.lax.Precision.DEFAULT)
  lanczos_fact = _generate_lanczos_factorization(jax)

  @functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
  def implicitly_restarted_lanczos_method(
      matvec: Callable, args: List, initial_state: jax.ShapedArray,
      num_krylov_vecs: int, numeig: int, which: Text, tol: float, maxiter: int,
      precision: JaxPrecisionType
  ) -> Tuple[jax.ShapedArray, List[jax.ShapedArray], int]:
    """
    Implicitly restarted lanczos factorization of `matvec`. The routine
    finds the lowest `numeig` eigenvector-eigenvalue pairs of `matvec`
    by alternating between compression and re-expansion of an initial
    `num_krylov_vecs`-step Lanczos factorization.

    Note: The caller has to ensure that the dtype of the return value
    of `matvec` matches the dtype of the initial state. Otherwise jax
    will raise a TypeError.

    NOTE: Under certain circumstances, the routine can return spurious
    eigenvalues 0.0: if the Lanczos iteration terminated early
    (after numits < num_krylov_vecs iterations)
    and numeig > numits, then spurious 0.0 eigenvalues will be returned.

    References:
    http://emis.impa.br/EMIS/journals/ETNA/vol.2.1994/pp1-21.dir/pp1-21.pdf
    http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Args:
      matvec: A callable representing the linear operator.
      args: Arguments to `matvec`.  `matvec` is called with
        `matvec(x, *args)` with `x` the input array on which
        `matvec` should act.
      initial_state: An starting vector for the iteration.
      num_krylov_vecs: Number of krylov vectors of the lanczos factorization.
        numeig: The number of desired eigenvector-eigenvalue pairs.
      which: Which eigenvalues to target.
        Currently supported: `which = 'LR'` (largest real part).
      tol: Convergence flag. If the norm of a krylov vector drops below `tol`
        the iteration is terminated.
      maxiter: Maximum number of (outer) iteration steps.
      precision: jax.lax.Precision used within lax operations.

    Returns:
      jax.ShapedArray: Eigenvalues
      List: Eigenvectors
      int: Number of inner krylov iterations of the last lanczos
        factorization.
    """
    shape = initial_state.shape
    dtype = initial_state.dtype

    dim = np.prod(shape).astype(np.int32)
    num_expand = num_krylov_vecs - numeig
    #note: the second part of the cond is for testing purposes
    if num_krylov_vecs <= numeig < dim:
      raise ValueError(f"num_krylov_vecs must be between numeig <"
                       f" num_krylov_vecs <= dim = {dim},"
                       f" num_krylov_vecs = {num_krylov_vecs}")
    if numeig > dim:
      raise ValueError(f"number of requested eigenvalues numeig = {numeig} "
                       f"is larger than the dimension of the operator "
                       f"dim = {dim}")

    # initialize arrays
    Vm = jax.numpy.zeros(
        (num_krylov_vecs, jax.numpy.ravel(initial_state).shape[0]), dtype=dtype)
    alphas = jax.numpy.zeros(num_krylov_vecs, dtype=dtype)
    betas = jax.numpy.zeros(num_krylov_vecs - 1, dtype=dtype)

    # perform initial lanczos factorization
    Vm, alphas, betas, residual, norm, numits, ar_converged = lanczos_fact(
        matvec, args, initial_state, Vm, alphas, betas, 0, num_krylov_vecs, tol,
        precision)
    fm = residual.ravel() * norm
    # generate needed functions
    shifted_QR = _shifted_QR(jax)
    check_eigvals_convergence = _check_eigvals_convergence_eigh(jax)
    get_vectors = _get_vectors(jax)

    # sort_fun returns `num_expand` least relevant eigenvalues
    # (those to be projected out)
    if which == 'LA':
      sort_fun = jax.tree_util.Partial(_LA_sort(jax), num_expand)
    elif which == 'SA':
      sort_fun = jax.tree_util.Partial(_SA_sort(jax), num_expand)
    elif which == 'LM':
      sort_fun = jax.tree_util.Partial(_LM_sort(jax), num_expand)
    else:
      raise ValueError(f"which = {which} not implemented")

    it = 1  # we already did one lanczos factorization
    def outer_loop(carry):
      alphas, betas, Vm, fm, it, numits, ar_converged, _, _, = carry
      # pack into alphas and betas into tridiagonal matrix
      Hm = jax.numpy.diag(alphas) + jax.numpy.diag(betas, -1) + jax.numpy.diag(
          betas.conj(), 1)
      evals, _ = jax.numpy.linalg.eigh(Hm)
      shifts, _ = sort_fun(evals)
      # perform shifted QR iterations to compress lanczos factorization
      # Note that ||fk|| typically decreases as one iterates the outer loop
      # indicating that iram converges.
      # ||fk|| = \beta_m in reference above
      Vk, Hk, fk = shifted_QR(Vm, Hm, fm, shifts, numeig)
      # extract new alphas and betas
      alphas = jax.numpy.diag(Hk)
      betas = jax.numpy.diag(Hk, -1)
      alphas = alphas.at[numeig:].set(0.0)
      betas = betas.at[numeig-1:].set(0.0)

      beta_k = jax.numpy.linalg.norm(fk)
      Hktest = Hk[:numeig, :numeig]
      matnorm = jax.numpy.linalg.norm(Hktest)
      converged = check_eigvals_convergence(beta_k, Hktest, matnorm, tol)


      def do_lanczos(vals):
        Vk, alphas, betas, fk, _, _, _, _ = vals
        # restart
        Vm, alphas, betas, residual, norm, numits, ar_converged = lanczos_fact(
            matvec, args, jax.numpy.reshape(fk, shape), Vk, alphas, betas,
            numeig, num_krylov_vecs, tol, precision)
        fm = residual.ravel() * norm
        return [Vm, alphas, betas, fm, norm, numits, ar_converged, False]

      def cond_lanczos(vals):
        return vals[7]

      res = jax.lax.while_loop(cond_lanczos, do_lanczos, [
          Vk, alphas, betas, fk,
          jax.numpy.linalg.norm(fk), numeig, False,
          jax.numpy.logical_not(converged)
      ])

      Vm, alphas, betas, fm, norm, numits, ar_converged = res[0:7]

      out_vars = [
          alphas, betas, Vm, fm, it + 1, numits, ar_converged, converged, norm
      ]
      return out_vars

    def cond_fun(carry):
      it, ar_converged, converged = carry[4], carry[6], carry[7]
      return jax.lax.cond(
          it < maxiter, lambda x: x, lambda x: False,
          jax.numpy.logical_not(jax.numpy.logical_or(converged, ar_converged)))

    converged = False
    carry = [alphas, betas, Vm, fm, it, numits, ar_converged, converged, norm]
    res = jax.lax.while_loop(cond_fun, outer_loop, carry)
    alphas, betas, Vm = res[0], res[1], res[2]
    numits, ar_converged, converged = res[5], res[6], res[7]
    Hm = jax.numpy.diag(alphas) + jax.numpy.diag(betas, -1) + jax.numpy.diag(
        betas.conj(), 1)
    # FIXME (mganahl): under certain circumstances, the routine can still
    # return spurious 0 eigenvalues: if lanczos terminated early
    # (after numits < num_krylov_vecs iterations)
    # and numeig > numits, then spurious 0.0 eigenvalues will be returned
    Hm = (numits > jax.numpy.arange(num_krylov_vecs))[:, None] * Hm * (
        numits > jax.numpy.arange(num_krylov_vecs))[None, :]

    eigvals, U = jax.numpy.linalg.eigh(Hm)
    inds = sort_fun(eigvals)[1][:numeig]
    vectors = get_vectors(Vm, U, inds, numeig)
    return eigvals[inds], [
        jax.numpy.reshape(vectors[n, :], shape) for n in range(numeig)
    ], numits

  return implicitly_restarted_lanczos_method


def gmres_wrapper(jax: types.ModuleType):
  """
  Allows Jax (the module) to be passed in as an argument rather than imported,
  since doing the latter breaks the build. In addition, instantiates certain
  of the enclosed functions as concrete objects within a Dict, allowing them to
  be cached. This avoids spurious recompilations that would otherwise be
  triggered by attempts to pass callables into Jitted functions.

  The important function here is functions["gmres_m"], which implements
  GMRES. The other functions are exposed only for testing.

  Args:
  ----
  jax: The imported Jax module.

  Returns:
  -------
  functions: A namedtuple of functions:
    functions.gmres_m = gmres_m
    functions.gmres_residual = gmres_residual
    functions.gmres_krylov = gmres_krylov
    functions.gs_step = _gs_step
    functions.kth_arnoldi_step = kth_arnoldi_step
    functions.givens_rotation = givens_rotation
  """
  jnp = jax.numpy
  JaxPrecisionType = type(jax.lax.Precision.DEFAULT)
  def gmres_m(
      A_mv: Callable, A_args: Sequence, b: jax.ShapedArray, x0: jax.ShapedArray,
      tol: float, atol: float, num_krylov_vectors: int, maxiter: int,
      precision: JaxPrecisionType) -> Tuple[jax.ShapedArray, float, int, bool]:
    """
    Solve A x = b for x using the m-restarted GMRES method. This is
    intended to be called via jax_backend.gmres.

    Given a linear mapping with (n x n) matrix representation
        A = A_mv(*A_args) gmres_m solves
        Ax = b          (1)
    where x and b are length-n vectors, using the method of
    Generalized Minimum RESiduals with M iterations per restart (GMRES_M).

    Args:
      A_mv: A function v0 = A_mv(v, *A_args) where v0 and v have the same shape.
      A_args: A list of positional arguments to A_mv.
      b: The b in A @ x = b.
      x0: Initial guess solution.
      tol, atol: Solution tolerance to achieve,
        norm(residual) <= max(tol * norm(b), atol).
        tol is also used to set the threshold at which the Arnoldi factorization
        terminates.
      num_krylov_vectors: Size of the Krylov space to build at each restart.
      maxiter: The Krylov space will be repeatedly rebuilt up to this many
        times.
    Returns:
      x: The approximate solution.
      beta: Norm of the residual at termination.
      n_iter: Number of iterations at termination.
      converged: Whether the desired tolerance was achieved.
    """
    num_krylov_vectors = min(num_krylov_vectors, b.size)
    x = x0
    b_norm = jnp.linalg.norm(b)
    tol = max(tol * b_norm, atol)
    for n_iter in range(maxiter):
      done, beta, x = gmres(A_mv, A_args, b, x, num_krylov_vectors, x0, tol,
                            b_norm, precision)
      if done:
        break
    return x, beta, n_iter, done

  def gmres(A_mv: Callable, A_args: Sequence, b: jax.ShapedArray,
            x: jax.ShapedArray, num_krylov_vectors: int, x0: jax.ShapedArray,
            tol: float, b_norm: float,
            precision: JaxPrecisionType) -> Tuple[bool, float, jax.ShapedArray]:
    """
    A single restart of GMRES.

    Args:
      A_mv: A function `v0 = A_mv(v, *A_args)` where `v0` and
                 `v` have the same shape.
      A_args: A list of positional arguments to A_mv.
      b: The `b` in `A @ x = b`.
      x: Initial guess solution.
      tol: Solution tolerance to achieve,
      num_krylov_vectors : Size of the Krylov space to build.
    Returns:
      done: Whether convergence was achieved.
      beta: Magnitude of residual (i.e. the error estimate).
      x: The approximate solution.
    """
    r, beta = gmres_residual(A_mv, A_args, b, x)
    k, V, R, beta_vec = gmres_krylov(A_mv, A_args, num_krylov_vectors,
                                     x0, r, beta, tol, b_norm, precision)
    x = gmres_update(k, V, R, beta_vec, x0)
    done = k < num_krylov_vectors - 1
    return done, beta, x

  @jax.jit
  def gmres_residual(A_mv: Callable, A_args: Sequence, b: jax.ShapedArray,
                     x: jax.ShapedArray) -> Tuple[jax.ShapedArray, float]:
    """
    Computes the residual vector r and its norm, beta, which is minimized by
    GMRES.

    Args:
      A_mv: A function v0 = A_mv(v, *A_args) where v0 and
        v have the same shape.
      A_args: A list of positional arguments to A_mv.
      b: The b in A @ x = b.
      x: Initial guess solution.
    Returns:
      r: The residual vector.
      beta: Its magnitude.
    """
    r = b - A_mv(x, *A_args)
    beta = jnp.linalg.norm(r)
    return r, beta

  def gmres_update(k: int, V: jax.ShapedArray, R: jax.ShapedArray,
                   beta_vec: jax.ShapedArray,
                   x0: jax.ShapedArray) -> jax.ShapedArray:
    """
    Updates the solution in response to the information computed by the
    main GMRES loop.

    Args:
      k: The final iteration which was reached by GMRES before convergence.
      V: The Arnoldi matrix of Krylov vectors.
      R: The R factor in H = QR where H is the Arnoldi overlap matrix.
      beta_vec: Stores the Givens factors used to map H into QR.
      x0: The initial guess solution.
    Returns:
      x: The updated solution.
    """
    q = min(k, R.shape[1])
    y = jax.scipy.linalg.solve_triangular(R[:q, :q], beta_vec[:q])
    x = x0 + V[:, :q] @ y
    return x

  @functools.partial(jax.jit, static_argnums=(2, 8))
  def gmres_krylov(
      A_mv: Callable, A_args: Sequence, n_kry: int, x0: jax.ShapedArray,
      r: jax.ShapedArray, beta: float, tol: float, b_norm: float,
      precision: JaxPrecisionType
  ) -> Tuple[int, jax.ShapedArray, jax.ShapedArray, jax.ShapedArray]:
    """
    Builds the Arnoldi decomposition of (A, v), where v is the normalized
    residual of the current solution estimate. The decomposition is
    returned as V, R, where V is the usual matrix of Krylov vectors and
    R is the upper triangular matrix in H = QR, with H the usual matrix
    of overlaps.

    Args:
      A_mv: A function `v0 = A_mv(v, *A_args)` where `v0` and
        `v` have the same shape.
      A_args: A list of positional arguments to A_mv.
      n_kry: Size of the Krylov space to build; this is called
        num_krylov_vectors in higher level code.
      x0: Guess solution.
      r: Residual vector.
      beta: Magnitude of r.
      tol: Solution tolerance to achieve.
      b_norm: Magnitude of b in Ax = b.
    Returns:
      k: Counts the number of iterations before convergence.
      V: The Arnoldi matrix of Krylov vectors.
      R: From H = QR where H is the Arnoldi matrix of overlaps.
      beta_vec: Stores Q implicitly as Givens factors.
    """
    n = r.size
    err = beta
    v = r / beta

    # These will store the Givens rotations used to update the QR decompositions
    # of the Arnoldi matrices.
    # cos : givens[0, :]
    # sine: givens[1, :]
    givens = jnp.zeros((2, n_kry), dtype=x0.dtype)
    beta_vec = jnp.zeros((n_kry + 1), dtype=x0.dtype)
    beta_vec = jax.ops.index_update(beta_vec, jax.ops.index[0], beta)
    V = jnp.zeros((n, n_kry + 1), dtype=x0.dtype)
    V = jax.ops.index_update(V, jax.ops.index[:, 0], v)
    R = jnp.zeros((n_kry + 1, n_kry), dtype=x0.dtype)

    # The variable data for the carry call. Each iteration modifies these
    # values and feeds the results to the next iteration.
    k = 0
    gmres_variables = (k, V, R, beta_vec, err,  # < The actual output we need.
                       givens)                  # < Modified between iterations.
    gmres_constants = (tol, A_mv, A_args, b_norm, n_kry)
    gmres_carry = (gmres_variables, gmres_constants)
    # The 'x' input for the carry call. Each iteration will receive an ascending
    # loop index (from the jnp.arange) along with the constant data
    # in gmres_constants.

    def gmres_krylov_work(gmres_carry: GmresCarryType) -> GmresCarryType:
      """
      Performs a single iteration of gmres_krylov. See that function for a more
      detailed description.

      Args:
        gmres_carry: The gmres_carry from gmres_krylov.
      Returns:
        gmres_carry: The updated gmres_carry.
      """
      gmres_variables, gmres_constants = gmres_carry
      k, V, R, beta_vec, err, givens = gmres_variables
      tol, A_mv, A_args, b_norm, _ = gmres_constants

      V, H = kth_arnoldi_step(k, A_mv, A_args, V, R, tol, precision)
      R_col, givens = apply_givens_rotation(H[:, k], givens, k)
      R = jax.ops.index_update(R, jax.ops.index[:, k], R_col[:])

      # Update the residual vector.
      cs, sn = givens[:, k] * beta_vec[k]
      beta_vec = jax.ops.index_update(beta_vec, jax.ops.index[k], cs)
      beta_vec = jax.ops.index_update(beta_vec, jax.ops.index[k + 1], sn)
      err = jnp.abs(sn) / b_norm
      gmres_variables = (k + 1, V, R, beta_vec, err, givens)
      return (gmres_variables, gmres_constants)

    def gmres_krylov_loop_condition(gmres_carry: GmresCarryType) -> bool:
      """
      This function dictates whether the main GMRES while loop will proceed.
      It is equivalent to:
        if k < n_kry and err > tol:
          return True
        else:
          return False
      where k, n_kry, err, and tol are unpacked from gmres_carry.

      Args:
        gmres_carry: The gmres_carry from gmres_krylov.
      Returns:
        (bool): Whether to continue iterating.
      """
      gmres_constants, gmres_variables = gmres_carry
      tol = gmres_constants[0]
      k = gmres_variables[0]
      err = gmres_variables[4]
      n_kry = gmres_constants[4]

      def is_iterating(k, n_kry):
        return k < n_kry

      def not_converged(args):
        err, tol = args
        return err >= tol
      return jax.lax.cond(is_iterating(k, n_kry),   # Predicate.
                          not_converged,            # Called if True.
                          lambda x: False,          # Called if False.
                          (err, tol))               # Arguments to calls.

    gmres_carry = jax.lax.while_loop(gmres_krylov_loop_condition,
                                     gmres_krylov_work,
                                     gmres_carry)
    gmres_variables, gmres_constants = gmres_carry
    k, V, R, beta_vec, err, givens = gmres_variables
    return (k, V, R, beta_vec)

  VarType = Tuple[int, jax.ShapedArray, jax.ShapedArray, jax.ShapedArray,
                  float, jax.ShapedArray]
  ConstType = Tuple[float, Callable, Sequence, jax.ShapedArray, int]
  GmresCarryType = Tuple[VarType, ConstType]


  @functools.partial(jax.jit, static_argnums=(6,))
  def kth_arnoldi_step(
      k: int, A_mv: Callable, A_args: Sequence, V: jax.ShapedArray,
      H: jax.ShapedArray, tol: float,
      precision: JaxPrecisionType) -> Tuple[jax.ShapedArray, jax.ShapedArray]:
    """
    Performs the kth iteration of the Arnoldi reduction procedure.
    Args:
      k: The current iteration.
      A_mv, A_args: A function A_mv(v, *A_args) performing a linear
        transformation on v.
      V: A matrix of size (n, K + 1), K > k such that each column in
        V[n, :k+1] stores a Krylov vector and V[:, k+1] is all zeroes.
      H: A matrix of size (K, K), K > k with H[:, k] all zeroes.
    Returns:
      V, H: With their k'th columns respectively filled in by a new
        orthogonalized Krylov vector and new overlaps.
    """

    def _gs_step(
        r: jax.ShapedArray,
        v_i: jax.ShapedArray) -> Tuple[jax.ShapedArray, jax.ShapedArray]:
      """
      Performs one iteration of the stabilized Gram-Schmidt procedure, with
      r to be orthonormalized against {v} = {v_0, v_1, ...}.

      Args:
        r: The new vector which is not in the initially orthonormal set.
        v_i: The i'th vector in that set.
      Returns:
        r_i: The updated r which is now orthonormal with v_i.
        h_i: The overlap of r with v_i.
      """
      h_i = jnp.vdot(v_i, r, precision=precision)
      r_i = r - h_i * v_i
      return r_i, h_i

    v = A_mv(V[:, k], *A_args)
    v_new, H_k = jax.lax.scan(_gs_step, init=v, xs=V.T)
    v_norm = jnp.linalg.norm(v_new)
    r_new = v_new / v_norm
    #  Normalize v unless it is the zero vector.
    r_new = jax.lax.cond(v_norm > tol,
                         lambda x: x[0] / x[1],
                         lambda x: 0.*x[0],
                         (v_new, v_norm)
                         )
    H = jax.ops.index_update(H, jax.ops.index[:, k], H_k)
    H = jax.ops.index_update(H, jax.ops.index[k+1, k], v_norm)
    V = jax.ops.index_update(V, jax.ops.index[:, k+1], r_new)
    return V, H

####################################################################
# GIVENS ROTATIONS
####################################################################
  @jax.jit
  def apply_rotations(H_col: jax.ShapedArray, givens: jax.ShapedArray,
                      k: int) -> jax.ShapedArray:
    """
    Successively applies each of the rotations stored in givens to H_col.

    Args:
      H_col : The vector to be rotated.
      givens: 2 x K, K > k matrix of rotation factors.
      k     : Iteration number.
    Returns:
      H_col : The rotated vector.
    """
    rotation_carry = (H_col, 0, k, givens)

    def loop_condition(carry):
      i = carry[1]
      k = carry[2]
      return jax.lax.cond(i < k, lambda x: True, lambda x: False, 0)

    def apply_ith_rotation(carry):
      H_col, i, k, givens = carry
      cs = givens[0, i]
      sn = givens[1, i]
      H_i = cs * H_col[i] - sn * H_col[i + 1]
      H_ip1 = sn * H_col[i] + cs * H_col[i + 1]
      H_col = jax.ops.index_update(H_col, jax.ops.index[i], H_i)
      H_col = jax.ops.index_update(H_col, jax.ops.index[i + 1], H_ip1)
      return (H_col, i + 1, k, givens)

    rotation_carry = jax.lax.while_loop(loop_condition,
                                        apply_ith_rotation,
                                        rotation_carry)
    H_col = rotation_carry[0]
    return H_col

  @jax.jit
  def apply_givens_rotation(H_col: jax.ShapedArray, givens: jax.ShapedArray,
                            k: int) -> Tuple[jax.ShapedArray, jax.ShapedArray]:
    """
    Applies the Givens rotations stored in the vectors cs and sn to the vector
    H_col. Then constructs a new Givens rotation that eliminates H_col's
    k'th element, yielding the corresponding column of the R in H's QR
    decomposition. Returns the new column of R along with the new Givens
    factors.

    Args:
      H_col : The column of H to be rotated.
      givens: A matrix representing the cosine and sine factors of the
        previous GMRES Givens rotations, in that order
        (i.e. givens[0, :] -> the cos factor).
      k     : Iteration number.
    Returns:
      R_col : The column of R obtained by transforming H_col.
      givens_k: The new elements of givens that zeroed out the k+1'th element
        of H_col.
    """
    # This call successively applies each of the
    # Givens rotations stored in givens[:, :k] to H_col.
    H_col = apply_rotations(H_col, givens, k)

    cs_k, sn_k = givens_rotation(H_col[k], H_col[k + 1])
    givens = jax.ops.index_update(givens, jax.ops.index[0, k], cs_k)
    givens = jax.ops.index_update(givens, jax.ops.index[1, k], sn_k)

    r_k = cs_k * H_col[k] - sn_k * H_col[k + 1]
    R_col = jax.ops.index_update(H_col, jax.ops.index[k], r_k)
    R_col = jax.ops.index_update(R_col, jax.ops.index[k + 1], 0.)
    return R_col, givens

  @jax.jit
  def givens_rotation(v1: float, v2: float) -> Tuple[float, float]:
    """
    Given scalars v1 and v2, computes cs = cos(theta) and sn = sin(theta)
    so that   [cs  -sn]  @ [v1] = [r]
              [sn   cs]    [v2]   [0]
    Args:
      v1, v2: The scalars.
    Returns:
      cs, sn: The rotation factors.
    """
    t = jnp.sqrt(v1**2 + v2**2)
    cs = v1 / t
    sn = -v2 / t
    return cs, sn

  fnames = [
      "gmres_m", "gmres_residual", "gmres_krylov",
      "kth_arnoldi_step", "givens_rotation"
  ]
  functions = [
      gmres_m, gmres_residual, gmres_krylov, kth_arnoldi_step,
      givens_rotation
  ]

  class Functions:

    def __init__(self, fun_dict):
      self.dict = fun_dict

    def __getattr__(self, name):
      return self.dict[name]

  return Functions(dict(zip(fnames, functions)))
