from typing import Callable, Sequence, Tuple
import jax
import jax.numpy as jnp
import jitted_functions
import functools

arnoldi_f = jitted_functions._generate_arnoldi_factorization(jax)
def gmres_m(A_mv: Callable, A_args: Sequence,
            b: jax.ShapedArray, x0: jax.ShapedArray, tol: float, atol: float,
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
  converged = False
  r, beta = gmres_residual(A_mv, A_args, b, x)
  b_norm = jnp.linalg.norm(b)
  for n_iter in range(maxiter):
    # pylint: disable=too-many-function-args
    x = gmres(A_mv, A_args, num_krylov_vectors, x, r, beta, tol, b_norm)
    r, beta = gmres_residual(A_mv, A_args, b, x)
    if beta <= max(tol*b_norm, atol):
      converged = True
      break
  return (x, beta, n_iter, converged)


#@jax.jit
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
  n = r.size
  errors = [beta / b_norm]
  v = r / beta

  # implicit matrix of Givens rotations
  sn = jnp.zeros(n_kry)
  cs = jnp.zeros(n_kry)

  Q = jnp.zeros(n, n_kry + 1)
  H = jnp.zeros(n_kry + 1, n_kry + 1)
  beta_vec = jnp.zeros(n_kry + 1)
  beta_vec = jax.ops.index_update(beta_vec, jax.ops.index[0], beta)
  for k in range(n_kry):
    # run Arnoldi
    Q, H, _, _ = arnoldi_f(A_mv, A_args, v, Q, H, k, 1, tol)
    Q = jax.ops.index_update(Q, jax.ops.index[:, k+1], Q_row[:])

    # eliminate the last element in H ith row and update the rotation matrix
    H_row, cs_k, sn_k = apply_givens(H[1:k+1, k], cs, sn, k)
    H = jax.ops.index_update(H, jax.ops.index[1:k+1, k], H_row[:])
    cs = jax.ops.index_update(cs, jax.ops.index[k], cs_k)
    sn = jax.ops.index_update(cs, jax.ops.index[k], sn_k)

    # update the residual vector
    beta_sn = -sn_k * beta_vec[k]
    beta_vec = jax.ops.index_update(beta_vec, jax.ops.index[k + 1], beta_sn)
    beta_cs = cs_k * beta_vec[k]
    beta_vec = jax.ops.index_update(beta_vec, jax.ops.index[k], beta_cs)
    err = jnp.abs(beta_sn) / b_norm
    errors.append(err)

    if (err <= tol):
      break
  y = jax.scipy.linalg.solve_triangular(H[:k, :k], beta[1:k])
  x = x0 + Q[:, :k] @ y
  return x


  v = r / beta
  Vk_1 = jnp.zeros((n_kry + 1, v.size), dtype=v.dtype)
  Htilde = jnp.zeros((n_kry + 1, n_kry + 1), dtype=v.dtype)
  arnoldi_f = jitted_functions._generate_arnoldi_factorization(jax)
  Vk_1, Htilde, _, _ = arnoldi_f(A_mv, A_args, v, Vk_1, Htilde, 0, n_kry, tol)
  Vk_1 = Vk_1.T
  Htilde = Htilde[:, :-1]
  Q, Rtilde = jnp.linalg.qr(Htilde, mode="complete")
  Q = Q.T.conj()
  R = Rtilde[:-1, :]
  g = beta*jnp.ravel(Q[:-1, 0])
  y = jax.scipy.linalg.solve_triangular(R, g)
  update = Vk_1[:, :-1] @ y
  x = x0 + update
  return x
#jit
#  def gmres(A_mv: Callable, A_args: Sequence, n_kry: int,
#            x0: jax.ShapedArray, r: jax.ShapedArray,
#            beta: float, tol: float) -> jax.ShapedArray:
#    """
#    Solve A x = b for x by the unrestarted GMRES method.
#    Given A, a trial solution x, the residual r,
#    and the size n_kry of the Krylov space, iterates x towards the solution,
#    by finding y in x = x_0 + V y minimizing ||beta - H y||.
#    """
#    v = r / beta
#    Vk_1 = jnp.zeros((n_kry + 1, v.size), dtype=v.dtype)
#    Htilde = jnp.zeros((n_kry + 1, n_kry + 1), dtype=v.dtype)
#    arnoldi_f = jitted_functions._generate_arnoldi_factorization(jax)
#    Vk_1, Htilde, _, _ = arnoldi_f(A_mv, A_args, v, Vk_1, Htilde, 0, n_kry, tol)
#    Vk_1 = Vk_1.T
#    Htilde = Htilde[:, :-1]
#    Q, Rtilde = jnp.linalg.qr(Htilde, mode="complete")
#    Q = Q.T.conj()
#    R = Rtilde[:-1, :]
#    g = beta*jnp.ravel(Q[:-1, 0])
#    y = jax.scipy.linalg.solve_triangular(R, g)
#    update = Vk_1[:, :-1] @ y
#    x = x0 + update
#    return x
