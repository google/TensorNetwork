from functools import partial
import numpy as np
from typing import Any, Callable, List
Tensor = Any


def _generate_jitted_eigsh_lanczos(jax):
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
  an input vector. `matvec` will be called as `matvec(*args, init)`.
  `init`: An initial input state to `matvec`.
  `ncv`: Number of krylov iterations (i.e. dimension of the Krylov space).
  `neig`: Number of eigenvalue-eigenvector pairs to be computed.
  `landelta`: Convergence parameter: if the norm of the current Lanczos vector
    falls below `landelta`, iteration is stopped.
  `reortho`: If `True`, reorthogonalize all krylov vectors at each step. 
     This should be used if `neig>1`.
  
  Args:
    jax: The `jax` module.
  Returns:
    Callable: A jitted function that does a lanczos iteration.
  
  """

  @partial(jax.jit, static_argnums=(3, 4, 5, 6))
  def jax_lanczos(matvec, arguments, init, ncv, neig, landelta, reortho):
    """
    Jitted lanczos routine.
    Args:
      matvec: A callable implementing the matrix-vector product of a 
        linear operator.
      arguments: Arguments to `matvec` additional to an input vector. 
        `matvec` will be called as `matvec(*args, init)`.
      init: An initial input state to `matvec`.
      ncv: Number of krylov iterations (i.e. dimension of the Krylov space).
      neig: Number of eigenvalue-eigenvector pairs to be computed.
      landelta: Convergence parameter: if the norm of the current Lanczos vector
        falls below `landelta`, iteration is stopped.
      reortho: If `True`, reorthogonalize all krylov vectors at each step. 
        This should be used if `neig>1`.
    Returns:
      list: Eigen values
      list: Eigen values
    """

    def body_modified_gram_schmidt(i, vals):
      vector, krylov_vectors = vals
      v = krylov_vectors[i, :]
      vector -= jax.numpy.vdot(v, jax.numpy.ravel(vector)) * jax.numpy.reshape(
          v, vector.shape)
      return [vector, krylov_vectors]

    def body_lanczos(vals):
      current_vector, krylov_vectors, vector_norms = vals[0:3]
      diagonal_elements, matvec, args, _ = vals[3:7]
      threshold, i, maxiteration = vals[7:]
      norm = jax.numpy.linalg.norm(jax.numpy.ravel(current_vector))
      normalized_vector = current_vector / norm
      normalized_vector, krylov_vectors = jax.lax.cond(
          reortho, True,
          lambda x: jax.lax.fori_loop(0, i, body_modified_gram_schmidt,
                                      [normalized_vector, krylov_vectors]),
          False, lambda x: [normalized_vector, krylov_vectors])
      Av = matvec(*args, normalized_vector)
      #Av = matvec(normalized_vector, args)

      diag_element = jax.numpy.dot(
          jax.numpy.conj(jax.numpy.ravel(normalized_vector)),
          jax.numpy.ravel(Av))

      res = jax.numpy.reshape(
          jax.numpy.ravel(Av) -
          jax.numpy.ravel(normalized_vector) * diag_element -
          krylov_vectors[i - 1] * norm, Av.shape)
      krylov_vectors = jax.ops.index_update(krylov_vectors, jax.ops.index[i, :],
                                            jax.numpy.ravel(normalized_vector))

      vector_norms = jax.ops.index_update(vector_norms, jax.ops.index[i - 1],
                                          norm)
      diagonal_elements = jax.ops.index_update(diagonal_elements,
                                               jax.ops.index[i - 1],
                                               diag_element)

      return [
          res, krylov_vectors, vector_norms, diagonal_elements, matvec, args,
          norm, threshold, i + 1, maxiteration
      ]

    def cond_fun(vals):
      _, _, _, _, _, _, norm, threshold, iteration, maxiteration = vals

      def check_thresh(check_vals):
        val, thresh = check_vals
        return jax.lax.cond(val < thresh, False, lambda x: x, True, lambda x: x)

      return jax.lax.cond(iteration <= maxiteration, [norm, threshold],
                          check_thresh, False, lambda x: x)

    numel = jax.numpy.prod(init.shape)
    krylov_vecs = jax.numpy.zeros((ncv + 1, numel), dtype=init.dtype)
    norms = jax.numpy.zeros(ncv, dtype=init.dtype)
    diag_elems = jax.numpy.zeros(ncv, dtype=init.dtype)

    norms = jax.ops.index_update(norms, jax.ops.index[0], 1.0)
    norms_dtype = np.real(init.dtype).dtype
    initvals = [
        init, krylov_vecs, norms, diag_elems, matvec, arguments,
        norms_dtype.type(1.0), landelta, 1, ncv
    ]
    output = jax.lax.while_loop(cond_fun, body_lanczos, initvals)
    final_state, krylov_vecs, norms, diags, _, _, _, _, it, _ = output
    krylov_vecs = jax.ops.index_update(krylov_vecs, jax.ops.index[it, :],
                                       jax.numpy.ravel(final_state))

    A_tridiag = jax.numpy.diag(diags) + jax.numpy.diag(
        norms[1:], 1) + jax.numpy.diag(jax.numpy.conj(norms[1:]), -1)
    eigvals, U = jax.numpy.linalg.eigh(A_tridiag)
    eigvals = eigvals.astype(A_tridiag.dtype)

    def body_vector(i, vals):
      krv, unitary, states = vals
      dim = unitary.shape[1]
      n, m = jax.numpy.divmod(i, dim)
      states = jax.ops.index_add(states, jax.ops.index[n, :],
                                 krv[m + 1, :] * unitary[m, n])
      return [krv, unitary, states]

    state_vectors = jax.numpy.zeros([neig, numel], dtype=init.dtype)
    _, _, vectors = jax.lax.fori_loop(0, neig * (krylov_vecs.shape[0] - 1),
                                      body_vector,
                                      [krylov_vecs, U, state_vectors])

    return jax.numpy.array(eigvals[0:neig]), [
        jax.numpy.reshape(vectors[n, :], init.shape) /
        jax.numpy.linalg.norm(vectors[n, :]) for n in range(neig)
    ]

  return jax_lanczos


def _generate_jitted_implicitly_rerstarted_arnoldi(jax):
  """
  """

  @jax.jit
  def modified_gram_schmidt_step_arnoldi(j: int, vals: List):
    """
    single step of a modified gram-schmidt orthogonalization
    """
    vector, krylov_vectors, n, H = vals
    v = krylov_vectors[j, :]
    h = jax.numpy.vdot(v, jax.numpy.ravel(vector))
    H = index_update(H, index[j, n], h)
    vector = vector - h * v
    return [vector, krylov_vectors, n, H]

  @partial(jax.jit, static_argnums=(5, 6, 7))
  def _arnoldi_factorization(matvec: Callable, args: List, v0: Tensor,
                             krylov_vectors: Tensor, H: Tensor, start: int,
                             num_krylov_vecs: int, eps: float):
    """
    Compute an arnoldi factorization of `matvec`.
    Args:
      matvec: The matrix vector product.
      args: List of arguments to `matvec`.
      v0: Initial state to `matvev`.
      krylov_vectors: An array for storing the krylov vectors. The individual
        vectors are stored as columns.
      H: Matrix of overlaps.
      start: Integer denoting the start position where the first produced krylov_vector 
        should be inserted into `krylov_vectors`
      num_krylov_vecs: Number of krylov iterations, should be identical to 
        `krylov_vectors.shape[0]`
    Returns:
      kv: An array of krylov vectors
      H: A matrix of overlaps 
      it: The number of performed iterations.
    """
    Z = jax.numpy.linalg.norm(v0)
    v = v0 / Z
    krylov_vectors = index_update(krylov_vectors, index[start, :], v)
    H = jax.lax.cond(start > 0, start,
                     lambda x: index_update(H, index[x, x - 1], Z), None,
                     lambda x: H)

    def body(vals):
      krylov_vectors, H, matvec, vector, _, threshold, i, maxiter = vals
      Av = matvec(*args, vector)
      initial_vals = [Av, krylov_vectors, i, H]
      Av, krylov_vectors, _, H = jax.lax.fori_loop(
          0, i + 1, modified_gram_schmidt_step_arnoldi, initial_vals)
      norm = jax.numpy.linalg.norm(Av)
      Av /= norm
      H = index_update(H, index[i + 1, i], norm)

      def update_krylov_vecs(args):
        krylov_vecs, vector, pos = args
        return index_update(krylov_vecs, index[pos, :], Av)

      krylov_vectors = index_update(krylov_vectors, index[i + 1, :], Av)
      return [krylov_vectors, H, matvec, Av, norm, threshold, i + 1, maxiter]

    def cond_fun(vals):
      kv, _, _, _, norm, threshold, iteration, maxiter = vals

      def check_thresh(check_vals):
        val, thresh = check_vals
        return jax.lax.cond(val < thresh, False, lambda x: x, True, lambda x: x)

      return jax.lax.cond(iteration < maxiter, [norm, threshold], check_thresh,
                          False, lambda x: x)

    norms_dtype = np.real(v0.dtype).dtype
    kvfinal, Hfinal, _, _, norm, _, it, _ = jax.lax.while_loop(
        cond_fun, body, [
            krylov_vectors, H, matvec, v,
            norms_dtype.type(100000000000.0), eps, start, num_krylov_vecs
        ])
    return kvfinal, Hfinal, it, norm < eps

  #######################################################
  ########  NEW SORTING FUCTIONS ISERTED HERE  ##########
  #######################################################

  @partial(jax.jit, static_argnums=(1,))
  def LR_sort(evals, p):
    inds = np.argsort(jax.numpy.real(evals), kind='stable')[::-1]
    shifts = evals[inds][-p:]
    return shifts, inds

  @partial(jax.jit, static_argnums=(1,))
  def LM_sort(evals, p):
    inds = np.argsort(jax.numpy.abs(evals), kind='stable')[::-1]
    shifts = evals[inds][-p:]
    return shifts, inds

  ########################################################
  ########################################################
  ########################################################

  @partial(jax.jit, static_argnums=(3, 4, 5))
  def shifted_QR(Vm, Hm, fm, k, p, which):
    funs = [LR_sort, LM_sort]
    evals, _ = jax.np.linalg.eig(Hm)
    shifts, _ = funs[which](evals, p)
    #compress to k = numeig
    q = jax.numpy.zeros(Hm.shape[0])
    q = index_update(q, index[-1], 1)
    m = Hm.shape[0]

    for shift in shifts:
      Qj, Rj = jax.np.linalg.qr(Hm - shift * jax.np.eye(m))
      Hm = Qj.T.conj() @ Hm @ Qj
      Vm = Qj.T @ Vm
      q = q @ Qj

    fk = Vm[k, :] * Hm[k, k - 1] + fm * q[k - 1]
    Vk = Vm[0:k, :]
    Hk = Hm[0:k, 0:k]
    H = jax.numpy.zeros((k + p + 1, k + p), dtype=fm.dtype)
    H = index_update(H, index[0:k, 0:k], Hk)
    Z = jax.np.linalg.norm(fk)
    v = fk / Z
    krylov_vectors = jax.numpy.zeros((k + p + 1, Vm.shape[1]), dtype=fm.dtype)
    krylov_vectors = index_update(krylov_vectors, index[0:k, :], Vk)
    krylov_vectors = index_update(krylov_vectors, index[k:], v)
    return krylov_vectors, H, fk

  @partial(jax.jit, static_argnums=(2,))
  def update_data(Vm_tmp, Hm_tmp, numits):
    Vm = Vm_tmp[0:numits, :]
    Hm = Hm_tmp[0:numits, 0:numits]
    fm = Vm_tmp[numits, :] * Hm_tmp[numits, numits - 1]
    return Vm, Hm, fm
