from functools import partial
from typing import List, Any, Tuple, Callable, Sequence
import numpy as np
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
      jax.numpy.ndarray: Eigenvalues
      list: Eigenvectors
    """

    def body_modified_gram_schmidt(i, vals):
      vector, krylov_vectors = vals
      v = krylov_vectors[i, :]
      vector -= jax.numpy.vdot(v, vector) * jax.numpy.reshape(v, vector.shape)
      return [vector, krylov_vectors]

    def body_lanczos(vals):
      current_vector, krylov_vectors, vector_norms = vals[0:3]
      diagonal_elements, matvec, args, _ = vals[3:7]
      threshold, i, maxiteration = vals[7:]
      norm = jax.numpy.linalg.norm(current_vector)
      normalized_vector = current_vector / norm
      normalized_vector, krylov_vectors = jax.lax.cond(
          reortho, True,
          lambda x: jax.lax.fori_loop(0, i, body_modified_gram_schmidt,
                                      [normalized_vector, krylov_vectors]),
          False, lambda x: [normalized_vector, krylov_vectors])
      Av = matvec(normalized_vector, *args)

      diag_element = jax.numpy.vdot(normalized_vector, Av)

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

  Note that the caller is responsible for dtype consistency between
  the inputs, i.e. dtypes between all input arrays have to match.

  Args:
    matvec: The matrix vector product. This function has to be wrapped into
      `jax.tree_util.Partial`. `matvec` will be called as `matvec(x, *args)`
      for an input vector `x`.
    args: List of arguments to `matvec`.
    v0: Initial state to `matvec`.
    krylov_vectors: An array for storing the krylov vectors. The individual
      vectors are stored as columns. The shape of `krylov_vecs` has to be
      (num_krylov_vecs + 1, np.ravel(v0).shape[0]).
    H: Matrix of overlaps. The shape has to be
      (num_krylov_vecs + 1,num_krylov_vecs + 1).
    start: Integer denoting the start position where the first
      produced krylov_vector should be inserted into `krylov_vectors`
    num_krylov_vecs: Number of krylov iterations, should be identical to
      `krylov_vectors.shape[0] + 1`
    eps: Convergence parameter. Iteration is terminated if the norm of a
      krylov-vector falls below `eps`.

  Returns:
    kv: An array of krylov vectors
    H: A matrix of overlaps
    it: The number of performed iterations.
    converged: Whether convergence was achieved.

  """

  @jax.jit
  def modified_gram_schmidt_step_arnoldi(j, vals):
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
    h = jax.numpy.vdot(v, vector)
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
    and `em` a cartesian basis vector of shape `(1, kv.shape[1])`
    with `em[0, -1] == 1` and 0 elsewhere.

    Note that the caller is responsible for dtype consistency between
    the inputs, i.e. dtypes between all input arrays have to match.

    Args:
      matvec: The matrix vector product.
      args: List of arguments to `matvec`.
      v0: Initial state to `matvec`.
      krylov_vectors: An array for storing the krylov vectors. The individual
        vectors are stored as columns.
        The shape of `krylov_vecs` has to be
        (num_krylov_vecs + 1, np.ravel(v0).shape[0]).
      H: Matrix of overlaps. The shape has to be
        (num_krylov_vecs + 1,num_krylov_vecs + 1).
      start: Integer denoting the start position where the first
        produced krylov_vector should be inserted into `krylov_vectors`
      num_krylov_vecs: Number of krylov iterations, should be identical to
        `krylov_vectors.shape[0] + 1`
      eps: Convergence parameter. Iteration is terminated if the norm of a
        krylov-vector falls below `eps`.
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

    # body of the arnoldi iteration
    def body(vals):
      krylov_vectors, H, matvec, vector, _, threshold, i, maxiter = vals
      Av = matvec(vector, *args)
      initial_vals = [Av, krylov_vectors, i, H]
      Av, krylov_vectors, _, H = jax.lax.fori_loop(
          0, i + 1, modified_gram_schmidt_step_arnoldi, initial_vals)
      norm = jax.numpy.linalg.norm(Av)
      Av /= norm
      H = jax.ops.index_update(H, jax.ops.index[i + 1, i], norm)
      krylov_vectors = jax.ops.index_update(krylov_vectors,
                                            jax.ops.index[i + 1, :],
                                            jax.numpy.ravel(Av))
      return [krylov_vectors, H, matvec, Av, norm, threshold, i + 1, maxiter]

    def cond_fun(vals):
      # Continue loop while iteration < num_krylov_vecs and norm > eps
      _, _, _, _, norm, _, iteration, _ = vals
      counter_done = (iteration >= num_krylov_vecs)
      norm_not_too_small = norm > eps
      continue_iteration = jax.lax.cond(counter_done,
                                        _, lambda x: False,
                                        _, lambda x: norm_not_too_small)

      return continue_iteration
    initial_norm = v.real.dtype.type(1.0+eps)
    initial_values = [krylov_vectors, H, matvec, v, initial_norm, eps, start,
                      num_krylov_vecs]
    final_values = jax.lax.while_loop(cond_fun, body, initial_values)
    kvfinal, Hfinal, _, _, norm, _, it, _ = final_values
    return kvfinal, Hfinal, it, norm < eps

  return _arnoldi_fact


def _implicitly_restarted_arnoldi(jax):
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
      which: Which eigenvalues to target. Currently supported: `which = 'LR'`
        or `which = 'LM'`.
      eps: Convergence flag. If the norm of a krylov vector drops below `eps`
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

  arnoldi_fact = _generate_arnoldi_factorization(jax)
  #######################################################
  ########  NEW SORTING FUCTIONS INSERTED HERE  #########
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

  @partial(jax.jit, static_argnums=(4, 5, 6))
  def shifted_QR(Vm, Hm, fm, evals, k, p, which):
    funs = [LR_sort, LM_sort]
    shifts, _ = funs[which](evals, p)
    #compress to k = numeig
    q = jax.numpy.zeros(Hm.shape[0])
    q = jax.ops.index_update(q, jax.ops.index[-1], 1)
    m = Hm.shape[0]

    for shift in shifts:
      Qj, _ = jax.numpy.linalg.qr(Hm - shift * jax.numpy.eye(m))
      Hm = Qj.T.conj() @ Hm @ Qj
      Vm = Qj.T @ Vm
      q = q @ Qj

    fk = Vm[k, :] * Hm[k, k - 1] + fm * q[k - 1]
    Vk = Vm[0:k, :]
    Hk = Hm[0:k, 0:k]
    H = jax.numpy.zeros((k + p + 1, k + p), dtype=fm.dtype)
    H = jax.ops.index_update(H, jax.ops.index[0:k, 0:k], Hk)
    Z = jax.numpy.linalg.norm(fk)
    v = fk / Z
    krylov_vectors = jax.numpy.zeros((k + p + 1, Vm.shape[1]), dtype=fm.dtype)
    krylov_vectors = jax.ops.index_update(krylov_vectors, jax.ops.index[0:k, :],
                                          Vk)
    krylov_vectors = jax.ops.index_update(krylov_vectors, jax.ops.index[k:], v)
    return krylov_vectors, H, fk

  @partial(jax.jit, static_argnums=(2,))
  def update_data(Vm_tmp, Hm_tmp, numits):
    Vm = Vm_tmp[0:numits, :]
    Hm = Hm_tmp[0:numits, 0:numits]
    fm = Vm_tmp[numits, :] * Hm_tmp[numits, numits - 1]
    return Vm, Hm, fm

  @partial(jax.jit, static_argnums=(3,))
  def get_vectors(Vm, unitary, inds, numeig):

    def body_vector(i, vals):
      krv, unitary, states, inds = vals
      dim = unitary.shape[1]
      n, m = jax.numpy.divmod(i, dim)
      states = jax.ops.index_add(states, jax.ops.index[n, :],
                                 krv[m, :] * unitary[m, inds[n]])
      return [krv, unitary, states, inds]

    state_vectors = jax.numpy.zeros([numeig, Vm.shape[1]], dtype=Vm.dtype)
    _, _, state_vectors, _ = jax.lax.fori_loop(
        0, numeig * Vm.shape[0], body_vector,
        [Vm, unitary, state_vectors, inds])
    state_norms = jax.numpy.linalg.norm(state_vectors, axis=1)
    state_vectors = state_vectors / state_norms[:, None]
    return state_vectors

  def implicitly_restarted_arnoldi_method(
      matvec, args, initial_state, num_krylov_vecs, numeig, which, eps,
      maxiter) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Implicitly restarted arnoldi factorization of `matvec`. The routine
    finds the lowest `numeig` eigenvector-eigenvalue pairs of `matvec`
    by alternating between compression and re-expansion of an initial
    `num_krylov_vecs`-step Arnoldi factorization.

    Note: The caller has to ensure that the dtype of the return value
    of `matvec` matches the dtype of the initial state. Otherwise jax
    will raise a TypeError.

    Args:
      matvec: A callable representing the linear operator.
      args: Arguments to `matvec`.  `matvec` is called with
        `matvec(x, *args)` with `x` the input array on which
        `matvec` should act.
      initial_state: An starting vector for the iteration.
      num_krylov_vecs: Number of krylov vectors of the arnoldi factorization.
        numeig: The number of desired eigenvector-eigenvalue pairs.
      which: Which eigenvalues to target. Currently supported: `which = 'LR'`
        or `which = 'LM'`.
      eps: Convergence flag. If the norm of a krylov vector drops below `eps`
        the iteration is terminated.
      maxiter: Maximum number of (outer) iteration steps.
    Returns:
      eta, U: Two lists containing eigenvalues and eigenvectors.
    """
    N = np.prod(initial_state.shape)
    p = num_krylov_vecs - numeig
    num_krylov_vecs = np.min([num_krylov_vecs, N])
    if (p <= 1) and (num_krylov_vecs < N):
      raise ValueError(f"`num_krylov_vecs` must be between `numeig` + 1 <"
                       f" `num_krylov_vecs` <= N={N},"
                       f" `num_krylov_vecs`={num_krylov_vecs}")

    dtype = initial_state.dtype
    # initialize arrays
    krylov_vectors = jax.numpy.zeros(
        (num_krylov_vecs + 1, jax.numpy.ravel(initial_state).shape[0]),
        dtype=dtype)
    H = jax.numpy.zeros((num_krylov_vecs + 1, num_krylov_vecs), dtype=dtype)

    # perform initial arnoldi factorization
    Vm_tmp, Hm_tmp, numits, converged = arnoldi_fact(matvec, args,
                                                     initial_state,
                                                     krylov_vectors, H, 0,
                                                     num_krylov_vecs, eps)
    # obtain an m-step arnoldi factorization
    Vm, Hm, fm = update_data(Vm_tmp, Hm_tmp, numits)

    it = 0
    if which == 'LR':
      _which = 0
    elif which == 'LM':
      _which = 1
    else:
      raise ValueError(f"which = {which} not implemented")
    # make sure the dtypes are matching
    if maxiter > 0:
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

    while (it < maxiter) and (not converged):
      evals, _ = jax.numpy.linalg.eig(Hm)
      krylov_vectors, H, fk = shifted_QR(Vm, Hm, fm, evals, numeig, p, _which)
      v0 = jax.numpy.reshape(fk, initial_state.shape)
      # restart
      Vm_tmp, Hm_tmp, _, converged = arnoldi_fact(matvec, args, v0,
                                                  krylov_vectors, H, numeig,
                                                  num_krylov_vecs, eps)
      Vm, Hm, fm = update_data(Vm_tmp, Hm_tmp, num_krylov_vecs)
      it += 1

    ev_, U_ = np.linalg.eig(np.array(Hm))
    eigvals = jax.numpy.array(ev_)
    U = jax.numpy.array(U_)
    _, inds = LR_sort(eigvals, _which)
    vectors = get_vectors(Vm, U, inds, numeig)

    return eigvals[inds[0:numeig]], [
        jax.numpy.reshape(vectors[n, :], initial_state.shape)
        for n in range(numeig)
    ]

  return implicitly_restarted_arnoldi_method


def gmres_wrapper(jax):
  """
  Allows Jax (the module) to be passed in as an argument rather than imported,
  since doing the latter breaks the build.
  Args:
    jax: The imported Jax module.
  Returns:
    gmres: A function performing gmres_m as described below.
  """
  jnp = jax.numpy
  functions = dict()

  @jax.jit
  def null_op(argument):
    return argument

  def gmres_m(A_mv: Callable, A_args: Sequence,
              b: jax.ShapedArray, x0: jax.ShapedArray, tol: float,
              num_krylov_vectors: int,
              maxiter: int) -> Tuple[jax.ShapedArray, float, int, bool]:
    """
    Solve A x = b for x using the m-restarted GMRES method. This is
    intended to be called via jax_backend.gmres.

    Given a linear mapping with (n x n) matrix representation
        A = A_mv(*A_args) gmres_m solves
        Ax = b          (1)
    where x and b are length-n vectors, using the method of
    Generalized Minimum RESiduals with M iterations per restart (GMRES_M).

    Args:

    A_mv     : A function `v0 = A_mv(v, *A_args)` where `v0` and
               `v` have the same shape.
    A_args   : A list of positional arguments to A_mv.
    b        : The `b` in `A @ x = b`.
    x0       : Initial guess solution.
    tol, atol: Solution tolerance to achieve,
               norm(residual) <= max(tol*norm(b), atol).
               tol is also used to set the threshold at which the Arnoldi
               factorization terminates.
    num_krylov_vectors
             : Size of the Krylov space to build at each restart.
    maxiter  : The Krylov space will be repeatedly rebuilt up to this many
               times.


    RETURNS
    -------
    x (array, (n,)) : The approximate solution.
    beta (float)    : Norm of the residual at termination.
    n_iter (int)    : Number of iterations at termination.
    converged (bool): Whether the desired tolerance was achieved.
    """
    x = x0
    b_norm = jnp.linalg.norm(b)
    n_iter = 1
    r, beta = gmres_residual(A_mv, A_args, b, x)
    x, _ = gmres(A_mv, A_args, num_krylov_vectors, x, r, beta, tol, b_norm)
    done = False
    for n_iter in range(2, maxiter + 1):
      # pylint: disable=too-many-function-args
      r, beta = gmres_residual(A_mv, A_args, b, x)
      x, done = gmres(A_mv, A_args, num_krylov_vectors, x, r, beta, tol, b_norm)
      if done:
        break
    return (x, beta, n_iter, done)


  @jax.jit
  def gmres_residual(A_mv: Callable, A_args: Sequence, b: jax.ShapedArray,
                     x: jax.ShapedArray) -> Tuple[jax.ShapedArray, float]:
    """
    Computes the residual vector r and its norm, beta, which is minimized by
    GMRES.
    """
    r = b - A_mv(x, *A_args)
    beta = jnp.linalg.norm(r)
    return r, beta


  @partial(jax.jit, static_argnums=(2, 6))
  def gmres_krylov(A_mv: Callable, A_args: Sequence, n_kry: int,
                   x0: jax.ShapedArray, r: jax.ShapedArray,
                   beta: float, tol: float, b_norm: float) -> jax.ShapedArray:
    """
    Solve A x = b for x by the unrestarted GMRES method.
    Given A, a trial solution x, the residual r,
    and the size n_kry of the Krylov space, iterates x towards the solution,
    by finding y in x = x_0 + V y minimizing ||beta - H y||.
    """
    done = False
    n = r.size
    err = beta / b_norm
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
    k = 0


    # The variable data for the carry call. Each iteration modifies these
    # values and feeds the results to the next iteration.
    gmres_variables = (k, V, R, beta_vec, err,  # The actual output we need.
                       givens)                     # Modified with iterations.

    gmres_constants = (tol, A_mv, A_args, b_norm)
    gmres_carry = (gmres_variables, gmres_constants)
    # The 'x' input for the carry call. Each iteration will receive an ascending
    # loop index (from the jnp.arange) along with the constant data
    # in gmres_constants.
    gmres_carry, _ = jax.lax.scan(gmres_scan,  # Called at each step.
                                  gmres_carry, # Changed, `carried` to next.
                                  xs=None,
                                  length=n_kry)# Redundant number of iterations.
    gmres_variables, gmres_constants = gmres_carry
    k, V, R, beta_vec, err, givens = gmres_variables
    done = k < n_kry - 1
    return (done, k, V, R, beta_vec)

  def gmres_update(k, V, R, beta_vec, x0):
    y = jax.scipy.linalg.solve_triangular(R[:k, :k], beta_vec[:k])
    x = x0 + V[:, :k] @ y
    return x

  def gmres(A_mv: Callable, A_args: Sequence, n_kry: int,
            x0: jax.ShapedArray, r: jax.ShapedArray,
            beta: float, tol: float, b_norm: float) -> jax.ShapedArray:
    done, k, V, R, beta_vec = gmres_krylov(A_mv, A_args, n_kry, x0, r, beta,
                                           tol,
                                           b_norm)
    x = gmres_update(k, V, R, beta_vec, x0)
    return x, done


  @jax.jit
  def gmres_scan(gmres_carry, gmres_xs):
    """
    This function wraps the main GMRES loop for compatibility with
    jax.lax.scan. If the error estimate err has dropped beneath the
    convergence threshold, no work is done at this iteration, so that the
    loop effectively terminates. Otherwise we call the main loop
    `gmres_work'.

    Args
    ----
    gmres_carry: The `carry` argument to jax.lax.scan.
     A nested tuple (gmres_variables, gmres_constants), themselves containing:
     gmres_variables = (final_k, V, R, beta_vec, err, givens):
      -final_k: The integer loop index. It is incremented until err has
                converged
                beneath tol and then held constant.
      -V      : An (n, n_kry + 1) array that will be used to store Krylov
                vectors during the internal Arnoldi process. It should be
                initialized as zeroes apart from V[0, :], the initial normalized
                Krylov vector.
      -R      : An (n_kry + 1, n_kry) array that will be used to store the R
                factor in the QR decomposition of the matrix of overlaps
                obtained during the Arnoldi process. It should be initialized
                as zeroes.
      -beta_vec: An (n_kry + 1) array that will be used to store the residual
                 between the solution vector and its projection into the Krylov
                 space. It should be initialized as zeroes apart from
                 beta_vec[0], the magnitude of the initial Krylov vector before
                 it was normalized.
      -err    : A real float that estimates the relativie magnitude of this
                residual at each iteration (the magnitude divided by that of
                the b in Ax = b).
      -givens : A (2, n_kry + 1) array that will be used to store the Givens
                rotation factors used to compute R (the upper triangular factor
                in the QR factorization of H) from H (the matrix of overlaps
                obtained during the Arnoldi process). It should be initialized
                as zeroes.
    gmres_constants:
      -tol  : The scan effectively terminates when err < tol by making
              subsequent iterations null ops.
      -A_mv : The linear operator representing A in A x = b.
      -A_args: A_mv is called as A_mv(*A_args, v).
      -b_norm: Norm of the b in A x = b.

    Returns
    -------
    The same entries as in gmres_carry, updated as described above.
    """
    _ = gmres_xs
    gmres_variables, gmres_constants = gmres_carry
    err = gmres_variables[4]
    tol = gmres_constants[0]
    gmres_carry = jax.lax.cond(err < tol,
                               gmres_carry, null_op,
                               gmres_carry, gmres_work)

    return gmres_carry, None


  @jax.jit
  def gmres_work(gmres_carry):
    """
    Performs one iteration of the main GMRES loop.
    Thus constructs the Arnoldi reduction of A_mv from v,
    A_mv -> V, H where V is a matrix of Krylov vectors and H stores
    the overlaps obtained during the orthogonalization of those vectors.
    The matrix H is used to obtain the residual of the projection of the
    solution onto this Krylov space. The code after the Arnoldi step
    builds the R matrix in the QR decomposition of H, and uses it to
    repeatedly update the solution, so that the Arnoldi process can
    be terminated early once convergence is reached.
    """
    gmres_variables, gmres_constants = gmres_carry
    k, V, R, beta_vec, err, givens = gmres_variables
    tol, A_mv, A_args, b_norm = gmres_constants

    V, H = kth_arnoldi_step(k, A_mv, A_args, V, R, tol)
    R_col, givens = apply_givens_rotation(H[:, k], givens, k)
    R = jax.ops.index_update(R, jax.ops.index[:, k], R_col[:])

    # Update the residual vector.
    cs, sn = givens[:, k] * beta_vec[k]
    beta_vec = jax.ops.index_update(beta_vec, jax.ops.index[k], cs)
    beta_vec = jax.ops.index_update(beta_vec, jax.ops.index[k + 1], sn)
    err = jnp.abs(sn) / b_norm
    gmres_variables = (k + 1, V, R, beta_vec, err, givens)
    return (gmres_variables, gmres_constants)


  @jax.jit
  def gs_step(r, v_i):
    """
    Performs one iteration of the stabilized Gram-Schmidt procedure, with
    r to be orthonormalized against {v} = {v_0, v_1, ...}.
    """
    h_i = jnp.vdot(v_i, r)
    r_i = r - h_i * v_i
    return r_i, h_i


  @partial(jax.jit, static_argnums=(5,))
  def kth_arnoldi_step(k, A_mv, A_args, V, H, tol):
    """
    Performs the kth iteration of the Arnoldi reduction procedure.
    Arguments
    ---------
    A_mv, A_args: A function A_mv(v, *A_args) performing a linear
                  transformation on v.
    V           : A matrix of size (n, k+1) such that each column in
                  V[n, :k+1] stores a Krylov vector and V[:, k+1] is all zeroes.
    H           : A matrix of size (k, k) with H[:, k] all zeroes.

    Returns
    -------
    V, H        : With their final columns filled in with a new orthogonalized
                  Krylov vector and new overlaps respectively.
    """
    v = A_mv(*A_args, V[:, k])
    #ks = itertools.repeat(k, V.shape[1])
    v_new, H_k = jax.lax.scan(gs_step, v, xs=V.T) # TODO: only nonzero steps
    v_norm = jnp.linalg.norm(v_new)
    r_new = v_new / v_norm
    #  Normalize v unless it is the zero vector.
    r_new = jax.lax.cond(v_norm > tol,
                         (v_new, v_norm), lambda x: x[0] / x[1],
                         v_new, lambda x: 0.*x,
                         )
    H = jax.ops.index_update(H, jax.ops.index[:, k], H_k)
    H = jax.ops.index_update(H, jax.ops.index[k+1, k], v_norm)
    V = jax.ops.index_update(V, jax.ops.index[:, k+1], r_new)
    return V, H


################################################################################
# GIVENS ROTATIONS
################################################################################

  @jax.jit
  def ith_rotation_work(tup):
    """
    This function does the work for apply_ith_rotation in the case that
    i < k. It finally applies the Givens rotation matrix about \theta stored as
    cos \theta = givens[0], sin \theta = givens[1] to the
    vector H_col.

    Args
    ----
    i     : The index of the rotation to be applied.
    givens: A vector of Givens rotation coefficients. The zeroth (first)
            component
            stores cos \theta (sin \theta).
    H_col : The vector to be rotated.

    Returns
    -------
    H_col : Having been rotated by the i'th element of Givens.
    """
    i, givens, H_col = tup
    cs = givens[0]
    sn = givens[1]
    H_i = cs * H_col[i] - sn * H_col[i + 1]
    H_ip1 = sn * H_col[i] + cs * H_col[i + 1]
    H_col = jax.ops.index_update(H_col, jax.ops.index[i], H_i)
    H_col = jax.ops.index_update(H_col, jax.ops.index[i + 1], H_ip1)
    return H_col


  @jax.jit
  def apply_ith_rotation(H_col_index, givens):
    """
    This function is a wrapper around ith_rotation_work that allows it to
    interface with jax.lax.scan, while also only applying it for
    to be applied for each i < k where k is an integer which is unknown at
    compile time. `scan` instead performs the iteration for each k, but
    due to the conditional enclosed here, the iterations for i >= k ar
    null ops.

    Args
    ----
    H_col_index: The `carry' argument to jax.lax.scan. A tuple H_col, i, k
                 where H_col is the vector to be rotated at the i'th iteration,
                 and k is the number of rotations to apply.
    givens      : The `xs` argument to jax.lax.scan. A (n_kry x 2) matrix of
                  Givens rotations, of which those in the  subblock
                  givens[:k, :]
                  are to be successively applied to H_col. Note givens is
                  transposed relative to its initialization in GMRES.

    Returns
    -------
    H_col_index : A new (H_col, i) where i has been incremented by 1 and
                  H_col has been rotated by each of givens[:k, :]
    """
    H_col, i, k = H_col_index

    H_col = jax.lax.cond(i < k,
                         (i, givens, H_col), ith_rotation_work,
                         H_col, null_op)

    return (H_col, i + 1, k), None


  @jax.jit
  def apply_rotations(tup):#H_col, givens, k):
    """
    Successively applies each of the rotations stored in givens
    to H_col.
    """
    H_col, givens, k = tup
    H_col_index, _ = jax.lax.scan(apply_ith_rotation, # Function to scan.
                                  (H_col, 0, k),      # Initial carried value.
                                  givens.T)        # Array to scan over.
    H_col, _, _ = H_col_index
    return H_col


  @jax.jit
  def apply_givens_rotation(H_col: jnp.array, givens: jnp.array, k: int):
    """
    Applies the Givens rotations stored in the vectors cs and sn to the vector
    H_col. Then constructs a new Givens rotation that eliminates H_col's
    k'th element, yielding the corresponding column of the R in H's QR
    decomposition. Returns the new column of R along with the new Givens
    factors.

    Arguments
    ---------
    H_col : The column of H to be rotated.
    givens: A matrix representing the cosine and sine factors of the
            previous GMRES Givens rotations, in that order
            (i.e. givens[0, :] -> the cos factor).

    Returns
    ------
    R_col : The column of R obtained by transforming H_col.
    givens_k: The new elements of Givens that zeroed out the k+1'th element of
              H_col.
    """
    # This call checks if k == 0, and if not successively applies each of the
    # Givens rotations stored in givens[:, :k] to H_col.
    H_col = jax.lax.cond(
        k == 0,                               # Condition.
        H_col, null_op,                       # Do nothing if k == 0.
        (H_col, givens, k), apply_rotations) # Else apply rotations.


    cs_k, sn_k = givens_rotation(H_col[k], H_col[k + 1])
    givens = jax.ops.index_update(givens, jax.ops.index[0, k], cs_k)
    givens = jax.ops.index_update(givens, jax.ops.index[1, k], sn_k)

    r_k = cs_k * H_col[k] - sn_k * H_col[k + 1]
    R_col = jax.ops.index_update(H_col, jax.ops.index[k], r_k)
    R_col = jax.ops.index_update(R_col, jax.ops.index[k + 1], 0.)
    return R_col, givens


  @jax.jit
  def givens_rotation(v1, v2):
    """
    Given scalars v1 and v2, computes cs = cos(theta) and sn = sin(theta)
    so that   [cs  -sn]  @ [v1] = [r]
              [sn   cs]    [v2]   [0]
    """
    t = jnp.sqrt(v1**2 + v2**2)
    cs = v1 / t
    sn = -v2 / t
    return cs, sn

  functions["null_op"] = null_op
  functions["gmres_m"] = gmres_m
  functions["gs_step"] = gs_step
  functions["kth_arnoldi_step"] = kth_arnoldi_step
  functions["givens_rotation"] = givens_rotation
  return functions
