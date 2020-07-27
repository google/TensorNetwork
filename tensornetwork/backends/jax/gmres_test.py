import tensorflow as tf
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
import pytest
from tensornetwork.backends.jax import jax_backend
import jax.config as config
import improved_gmres as gmres
from functools import partial
# pylint: disable=no-member
config.update("jax_enable_x64", True)

jax_qr_dtypes = [np.float64] #[np.float32, np.float64, np.complex64, np.complex128]
@pytest.mark.parametrize("dtype", jax_qr_dtypes)
def test_gmres_on_small_known_problem(dtype):
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype

  A = jax.numpy.array(([[1, 1], [3, -4]]), dtype=dtype)
  b = jax.numpy.array([3, 2], dtype=dtype)
  x0 = jax.numpy.ones(2, dtype=dtype)
  n_kry = 2
  maxiter = 1

  @jax.tree_util.Partial
  def A_mv(x):
    return A @ x
  tol = A.size*jax.numpy.finfo(dtype).eps
  x, _, _, _ = gmres.gmres_m(A_mv, [], b, x0, tol, n_kry, maxiter)
  solution = jax.numpy.array([2., 1.], dtype=dtype)
  np.testing.assert_allclose(x, solution, atol=tol)


@pytest.mark.parametrize("dtype", jax_qr_dtypes)
def test_gs(dtype):
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype
  n = 8
  A = np.zeros((n, 2), dtype=dtype)
  A[:-1, 0] = 1.0
  Ai = A[:, 0] / np.linalg.norm(A[:, 0])
  A[:, 0] = Ai
  A[-1, -1] = 1.0
  A = jnp.array(A)

  x0 = jnp.array(np.random.rand(n).astype(dtype))
  v_new, _ = jax.lax.scan(gmres._gs_step, x0, xs=A.T)
  dotcheck = v_new @ A
  tol = A.size*jax.numpy.finfo(dtype).eps
  np.testing.assert_allclose(dotcheck, np.zeros(2), atol=tol)


@pytest.mark.parametrize("dtype", jax_qr_dtypes)
def test_gmres_arnoldi_step(dtype):
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype
  n = 4
  n_kry = n
  np.random.seed(10)
  A = jnp.array(np.random.rand(n, n).astype(dtype))
  x0 = jnp.array(np.random.rand(n).astype(dtype))
  Q = np.zeros((n, n_kry + 1), dtype=x0.dtype)
  Q[:, 0] = x0/jnp.linalg.norm(x0)
  Q = jnp.array(Q)
  H = jnp.zeros((n_kry + 1, n_kry), dtype=x0.dtype)
  tol = A.size*jax.numpy.finfo(dtype).eps
  @jax.tree_util.Partial
  def A_mv(x):
    return A @ x
  for k in range(n_kry):
    Q, H = gmres.kth_arnoldi_step(k, A_mv, [], Q, H, tol)
  QAQ = Q[:, :n_kry].conj().T @ A @ Q[:, :n_kry]
  np.testing.assert_allclose(H[:n_kry, :], QAQ, atol=tol)


@pytest.mark.parametrize("dtype", jax_qr_dtypes)
def test_givens(dtype):
  np.random.seed(10)
  v = jnp.array(np.random.rand(2).astype(dtype))
  cs, sn = gmres.givens_rotation(*v)
  rot = np.zeros((2, 2))
  rot[0, 0] = cs
  rot[1, 1] = cs
  rot[0, 1] = -sn
  rot[1, 0] = sn
  rot = jnp.array(rot)
  result = rot @ v
  tol = 4*jax.numpy.finfo(dtype).eps
  np.testing.assert_allclose(result[-1], 0., atol=tol)
