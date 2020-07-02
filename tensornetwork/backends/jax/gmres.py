from functools import partial
from typing import Any, Optional, Tuple, Callable, List, Text, Type, Sequence
import jax
import jax.numpy as jnp

def gmres_m(A_mv: Callable, A_args: Sequence, 
            b: jnp.ShapedArray, x0: jnp.ShapedArray, tol: float, atol: float,
            num_krylov_vectors: int,
            maxiter: int) -> Tuple[jnp.ShapedArray, float, int, bool]:
  """
  Solve A x = b for x using the m-restarted GMRES method. This is
  intended to be called via jax_backend.gmres.

  Given a linear mapping with (n x n) matrix representation
      A = A_mv(*A_args) gmres_m solves
      Ax = b          (1)
  where x and b are length-b vectors, using the method of
  Generalized Minimum RESiduals with M iterations per restart (GMRES_M).

  Args:

  A_mv     : A function `v0 = A_mv(v, *A_args, **A_kwargs)` where `v0` and
             `v` have the same shape.
  b        : The `b` in `A @ x = b`.
  A_args   : Positional arguments to `A_mv`.
  x0       : Initial guess solution.
  tol, atol: Solution tolerance to achieve,
             norm(residual) <= max(tol*norm(b), atol).
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
  num_krylov_vectors += 1
  x = x0
  converged = False
  r, beta = gmres_residual(A_mv, A_args, b, x)
  b_norm = jnp.linalg.norm(b)
  for n_iter in range(maxiter):
    # pylint: disable=too-many-function-args
    x = gmres(A_mv, A_args, num_krylov_vectors, x, r, beta)
    r, beta = gmres_residual(A_mv, A_args, b, x)
    if beta <= max(tol*b_norm, atol):
      converged = True
      break
  return (x, beta, n_iter, converged)


@jax.jit
def gmres_residual(A_mv: Callable, A_args: Sequence, b: jnp.ShapedArray,
                   x: jnp.ShapedArray) -> Tuple[jnp.ShapedArray, float]:
  """
  Computes the residual vector r and its norm, beta, which is minimized by
  GMRES.
  """
  r = b - A_mv(x, *A_args)
  beta = jnp.linalg.norm(r)
  return r, beta


@partial(jax.jit, static_argnums=(2,))
def gmres(A_mv: Callable, A_args: Sequence, n_kry: int,
          x0: jnp.ShapedArray, r: jnp.ShapedArray,
          beta: float) -> jnp.ShapedArray:
  """
  Solve A x = b for x by the unrestarted GMRES method.
  Given A, a trial solution x, the residual r,
  and the size n_kry of the Krylov space, iterates x towards the solution,
  by finding y in x = x_0 + V y minimizing ||beta - H y||.
  """
  v = r / beta
  Vk_1, Htilde = gmres_arnoldi(A_mv, A_args, n_kry, v)
  Q, Rtilde = jnp.linalg.qr(Htilde, mode="complete")
  Q = Q.T.conj()
  R = Rtilde[:-1, :]
  g = beta*jnp.ravel(Q[:-1, 0])
  y = jax.scipy.linalg.solve_triangular(R, g)
  update = Vk_1[:, :-1] @ y
  x = x0 + update
  return x


@partial(jax.jit, static_argnums=(2,))
def gmres_arnoldi(A_mv: Callable,
                  A_args: Sequence,
                  n_kry: int,
                  v0: jnp.ShapedArray) -> Tuple[jnp.ShapedArray,
                                                jnp.ShapedArray]:
  """
  Given an (m x m) matrix A, a vector v0, and a dimension n_kry, finds
  an orthonormal basis on the order-(n_kry+1) Krylov space defined by
  (A, v0) as span{v0, A@v0, A^2@v0, .... A^(n_kry)@v0}.

  This orthonormal basis is found by the Arnoldi method: v0 is repeatedly
  multiplied by A, and each result orthogonalized against its ancestors
  using the stabilized Gram-Schmidt procedure. The basis vectors are
  stored in the columns of an (m x n_kry+1) matrix V.

  The Arnoldi process also generates an (n_kry+1, n_kry) upper-Hessenberg
  matrix H. The submatrix H[:-1, :] is the projection of A onto the
  first n_kry columns of V, H[:-1, :] = (V[:, :-1])^T @ A @ V[:, :-1].
  The last row of H is zero except for H[-1, -1] = norm(V[:, -1]).

  A is represented as a function y=A(*A_args, x) implementing a linear map
  y = A@x. This is possible because A is never modified during the procedure.

  Presently, this is redundant with the Arnoldi implementation used for
  eigensolving. Later, this function will be replaced with one that
  builds V and H as their QR decompositions.

  PARAMETERS
  ----------
  A_mv, Callable: Function A_mv(x, *A_args, **A_kwargs) returning y = A@x.
                  This must
                  be a Jax type, which can be achieved by wrapping it in
                  tree_utils.Partial.
  A_args, List  : List containing any arguments to A_mv besides x. Only
                  positional arguments are allowed, and each must be a
                  Jax type.
  n_kry, Int    : The dimensions of the Krylov subspace.
  v0 (N,) array : Vector defining the Krylov subspace.


  RETURNS
  -------
  V (m x n_kry+1) : Columns are an orthonormal basis of the Krylov
                    subspace.
  H (n_kry+1, n_kry:): Upper Hessenberg projection of A onto H, plus:w
                       a diagonal entry ||V[:, -1]||.
  """
  dtype = v0.dtype
  m = v0.shape[0]
  V = jnp.zeros((m, n_kry + 1), dtype=dtype)

  v = v0 / jnp.linalg.norm(v0)  # Normalize the input vector.
  V = jax.ops.index_update(V, jax.ops.index[:, 0], v)  # Use as first Krylov vec

  #@jax.jit
  def this_arnoldi_scan_function(carry, x):
      return _arnoldi_scan_function(carry, x, A_mv, A_args)
  carry, stack = jax.lax.scan(this_arnoldi_scan_function,
                              (V, v),
                              xs=jnp.arange(n_kry))
                              # jnp.arange(n_kry))
  V, _ = carry
  H, h_off = stack
  H = H.T + jnp.diag(h_off, -1)[:n_kry+1, :n_kry]
  return (V, H)


@jax.jit
def _arnoldi_scan_function(carry: Sequence, k: int, A_mv: Callable,
                           A_args: Sequence) -> Tuple[Sequence, Sequence]:
  """
  Main loop of arnoldi_krylov in a jax.lax.scan - friendly format.
  k is the current iteration index. carry is V, v; the second being
  the latest value of v. stack is hs, v_norm; v_norm is the first
  lower off diagonal of the eventual H, and hs is the upper triangle.
  """
  V, v_old = carry
  eps = jnp.finfo(v_old.dtype).eps
  r = A_mv(v_old, *A_args)
  v_new, hs = _gs_orthogonalize(V, r)
  v_norm = jnp.linalg.norm(v_new)
  switch = v_norm > eps

  #  Normalize v unless it is the zero vector.
  v_new = jax.lax.cond(switch,
                       (v_new, v_norm), lambda x: x[0] / x[1],
                       v_new, lambda x: jnp.zeros(x.size, dtype=x.dtype),
                       )
  V = jax.ops.index_update(V, jax.ops.index[:, k+1], v_new)
  newcarry = (V, v_new)
  stack = (hs, v_norm)
  return newcarry, stack


@jax.jit
def _gs_step(r: jnp.ShapedArray,
             v_i: jnp.ShapedArray) -> Tuple[jnp.ShapedArray, jnp.ShapedArray]:
  """
  Performs one iteration of the stabilized Gram-Schmidt procedure, with
  r to be orthonormalized against {v} = {v_0, v_1, ...}.
  """
  h_i = jnp.vdot(v_i, r)
  r_i = r - h_i * v_i
  return r_i, h_i


@jax.jit
def _gs_orthogonalize(V: jnp.ShapedArray,
                      r: jnp.ShapedArray) -> Tuple[jnp.ShapedArray,
                                                   jnp.ShapedArray]:
  """
  Orthonormalize r against the vectors in the columns of V using
  the stablized Gram-Schmidt procedure. More specifically, given
  V whose columns form an orthonormal basis {v0, v1, ...} and some
  other vector r, return r_new so that {v0, v1, ..., r_new} is an
  orthonormal basis on span{v0, v1, ..., r}.

  PARAMETERS
  ----------
  V, array-like (N, n): Columns are the basis vectors to be orthonormalized
                        against. They are assumed already orthonormal.
  r, array-like (N,)  : The vector to orthonormalize against V.


  RETURNS
  -------
  r_new, array-like (N,) : Orthonormal to the columns of  V, such that
                           {V, r_new} spans
                           the same space as {V, r}.
  hs, array-like (n,)  : Projections of the {v} onto successive r_new during
                         the procedure.
  """
  r_new, hs = jax.lax.scan(_gs_step, r, xs=V.T)
  return (r_new, hs)
