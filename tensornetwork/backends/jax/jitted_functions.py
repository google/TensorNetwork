from functools import partial
import numpy as np


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
  an input vector. `matvec` will be called as `matvec(init, *args)`.
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
        `matvec` will be called as `matvec(init, *args)`.
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
      Av = matvec(normalized_vector, *args)

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

    norms_dtype = jax.numpy.real(jax.numpy.empty((0, 0),
                                                 dtype=init.dtype)).dtype
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


def _generate_arnoldi_factorization(jax):
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

  The arguments to `_arnoldi_fact` are:

  Args:
    matvec: The matrix vector product. This function has to be wrapped into 
      `jax.tree_util.Partial`. `matvec` will be called as `matvec(x, *args)`
      for an input vector `x`.
    args: List of arguments to `matvec`.
    v0: Initial state to `matvec`.
    krylov_vectors: An array for storing the krylov vectors. The individual
      vectors are stored as columns.a
    H: Matrix of overlaps.
    start: Integer denoting the start position where the first 
      produced krylov_vector should be inserted into `krylov_vectors`
    num_krylov_vecs: Number of krylov iterations, should be identical to 
      `krylov_vectors.shape[0]`
  Returns:
    kv: An array of krylov vectors
    H: A matrix of overlaps 
    it: The number of performed iterations.

  """

  @jax.jit
  def modified_gram_schmidt_step_arnoldi(j: int, vals: List):
    """
    Single step of a modified gram-schmidt orthogonalization.
    Args:
      j: Integer value denoting the vector to be orthogonalized.
      vals: A list of variables:
        `vector`: The current vector to be orthogonalized
        to all previous ones
        `krylov_vectors`: jax.array of collected krylov vectors
        `n`: integer denoting the column-position of the overlap 
          <`krylov_vector`|`vector`> within `H`.
    Returns:
      updated vals.
             
    """
    vector, krylov_vectors, n, H = vals
    v = krylov_vectors[j, :]
    h = jax.numpy.vdot(v, jax.numpy.ravel(vector))
    H = jax.ops.index_update(H, jax.ops.index[j, n], h)
    vector = vector - h * jax.numpy.reshape(v, vector.shape)
    return [vector, krylov_vectors, n, H]

  @partial(jax.jit, static_argnums=(5, 6, 7))
  def _arnoldi_fact(matvec, args, v0, krylov_vectors, H, start, num_krylov_vecs,
                    eps):
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
    and `em` a kartesian basis vector of shape `(1, kv.shape[1])` 
    with `em[0, -1] == 1` and 0 elsewhere.

    Args:
      matvec: The matrix vector product.
      args: List of arguments to `matvec`.
      v0: Initial state to `matvec`.
      krylov_vectors: An array for storing the krylov vectors. The individual
        vectors are stored as columns.a
      H: Matrix of overlaps.
      start: Integer denoting the start position where the first 
        produced krylov_vector should be inserted into `krylov_vectors`
      num_krylov_vecs: Number of krylov iterations, should be identical to 
        `krylov_vectors.shape[0]`
    Returns:
      kv: An array of krylov vectors
      H: A matrix of overlaps 
      it: The number of performed iterations.
    """
    Z = jax.numpy.linalg.norm(v0)
    v = v0 / Z
    krylov_vectors = jax.ops.index_update(krylov_vectors,
                                          jax.ops.index[start, :],
                                          jax.numpy.ravel(v))
    H = jax.lax.cond(
        start > 0, start,
        lambda x: jax.ops.index_update(H, jax.ops.index[x, x - 1], Z), None,
        lambda x: H)

    def body(vals):
      krylov_vectors, H, matvec, vector, _, threshold, i, maxiter = vals
      Av = matvec(vector, *args)
      initial_vals = [Av, krylov_vectors, i, H]
      Av, krylov_vectors, _, H = jax.lax.fori_loop(
          0, i + 1, modified_gram_schmidt_step_arnoldi, initial_vals)
      norm = jax.numpy.linalg.norm(jax.numpy.ravel(Av))
      Av /= norm
      H = jax.ops.index_update(H, jax.ops.index[i + 1, i], norm)
      krylov_vectors = jax.ops.index_update(krylov_vectors,
                                            jax.ops.index[i + 1, :],
                                            jax.numpy.ravel(Av))
      return [krylov_vectors, H, matvec, Av, norm, threshold, i + 1, maxiter]

    def cond_fun(vals):
      _, _, _, _, norm, threshold, iteration, maxiter = vals

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

  return _arnoldi_fact
