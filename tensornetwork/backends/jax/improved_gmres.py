from typing import Callable, Sequence, Tuple
import jax
import jax.numpy as jnp
import jitted_functions
import functools
import itertools

arnoldi_f = jitted_functions._generate_arnoldi_factorization(jax)

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


def gmres(A_mv: Callable, A_args: Sequence, n_kry: int,
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
                     givens)                     # Modified between iterations.

  gmres_constants = (tol, A_mv, A_args, b_norm)
  gmres_carry = (gmres_variables, gmres_constants)
  # The 'x' input for the carry call. Each iteration will receive an ascending
  # loop index (from the jnp.arange) along with the constant data
  # in gmres_constants.
  gmres_carry, _ = jax.lax.scan(gmres_scan,  # Called at each step.
                                gmres_carry, # Changed, `carried` to next step.
                                xs=None,
                                length=n_kry)# Redundant number of iterations.
  gmres_variables, gmres_constants = gmres_carry
  k, V, R, beta_vec, err, givens = gmres_variables
  done = k < n_kry - 1

  y = jax.scipy.linalg.solve_triangular(R[:k, :k], beta_vec[:k])
  x = x0 + V[:, :k] @ y
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
    -final_k: The integer loop index. It is incremented until err has converged
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
  gmres_variables, gmres_constants = gmres_carry
  err = gmres_variables[4]
  tol = gmres_constants[0]
  gmres_carry = jax.lax.cond(err < tol,
                             gmres_carry, null_op,
                             gmres_carry, gmres_work)

  return gmres_carry, None


#@partial(jax.jit, static_argnums=(6,))
#def gmres_work(k, V, R, beta_vec, err, givens, tol, A_mv, A_args, b_norm):
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
def _gs_step(r, v_i):
  """
  Performs one iteration of the stabilized Gram-Schmidt procedure, with
  r to be orthonormalized against {v} = {v_0, v_1, ...}.
  """
  h_i = jnp.vdot(v_i, r)
  r_i = r - h_i * v_i
  return r_i, h_i


@functools.partial(jax.jit, static_argnums=(5,))
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
  v_new, H_k = jax.lax.scan(_gs_step, v, xs=V.T) # TODO: only do the nonzero steps
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
  givens: A vector of Givens rotation coefficients. The zeroth (first) component
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
                Givens rotations, of which those in the  subblock givens[:k, :] 
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
