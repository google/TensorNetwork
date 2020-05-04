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

    def body_reortho(i, vals):
      vector, krylov_vectors = vals
      v = krylov_vectors[i, :]
      vector -= jax.numpy.dot(jax.numpy.conj(v),
                              jax.numpy.ravel(vector)) * jax.numpy.reshape(
                                  v, vector.shape)
      return [vector, krylov_vectors]

    def body_lanczos(vals):
      current_vector, krylov_vectors, vector_norms = vals[0:3]
      diagonal_elements, matvec, args, _ = vals[3:7]
      threshold, i, maxiteration = vals[7:]
      #current_vector = krylov_vectors[i,:]
      norm = jax.numpy.linalg.norm(jax.numpy.ravel(current_vector))
      normalized_vector = current_vector / norm
      normalized_vector, krylov_vectors = jax.lax.cond(
          reortho, True, lambda x: jax.lax.fori_loop(
              0, i, body_reortho, [normalized_vector, krylov_vectors]), False,
          lambda x: [normalized_vector, krylov_vectors])
      Av = matvec(*args, normalized_vector)

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
    #pylint: disable=line-too-long
    final_state, krylov_vecs, norms, diags, _, _, _, _, it, _ = jax.lax.while_loop(
        cond_fun, body_lanczos, initvals)
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
      states = jax.ops.index_add(states, jax.ops.index[n],
                                 krv[m + 1, :] * unitary[m, n])
      return [krv, unitary, states]

    state_vector = jax.numpy.zeros([neig, numel], dtype=init.dtype)
    _, _, vector = jax.lax.fori_loop(0, neig * (krylov_vecs.shape[0] - 1),
                                     body_vector,
                                     [krylov_vecs, U, state_vector])
    vector /= jax.numpy.linalg.norm(vector)
    return jax.numpy.array(eigvals[0:neig]), [
        jax.numpy.reshape(vector[n, :], init.shape) /
        jax.numpy.linalg.norm(vector[n, :]) for n in range(neig)
    ]

  return jax_lanczos
