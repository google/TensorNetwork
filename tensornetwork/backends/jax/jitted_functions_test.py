import numpy as np
import jax
import pytest
#pylint: disable=line-too-long
from tensornetwork.backends.jax import jitted_functions
jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_arnoldi_factorization(dtype):
  np.random.seed(10)
  D = 20
  mat = np.random.rand(D, D).astype(dtype)
  x = np.random.rand(D).astype(dtype)

  @jax.tree_util.Partial
  @jax.jit
  def matvec(vector, matrix):
    return matrix @ vector

  arnoldi = jitted_functions._generate_arnoldi_factorization(jax)
  ncv = 40
  kv = jax.numpy.zeros((ncv + 1, D), dtype=dtype)
  H = jax.numpy.zeros((ncv + 1, ncv), dtype=dtype)
  start = 0
  kv, H, it, _ = arnoldi(matvec, [mat], x, kv, H, start, ncv, 0.01)
  Vm = jax.numpy.transpose(kv[:it, :])
  Hm = H[:it, :it]
  fm = kv[it, :] * H[it, it - 1]
  em = np.zeros((1, Vm.shape[1]))
  em[0, -1] = 1
  np.testing.assert_almost_equal(mat @ Vm - Vm @ Hm - fm[:, None] * em,
                                 np.zeros((it, Vm.shape[1])).astype(dtype))


jax_dtypes = [np.float32, np.float64, np.complex64, np.complex128]
@pytest.mark.parametrize("dtype", jax_dtypes)
def test_gmres_on_small_known_problem(dtype):
  """
  GMRES produces the correct result on an analytically solved
  linear system.
  """
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype
  gmres = jitted_functions.gmres_wrapper(jax)

  A = jax.numpy.array(([[1, 1], [3, -4]]), dtype=dtype)
  b = jax.numpy.array([3, 2], dtype=dtype)
  x0 = jax.numpy.ones(2, dtype=dtype)
  n_kry = 2
  maxiter = 1

  @jax.tree_util.Partial
  def A_mv(x):
    return A @ x
  tol = A.size*jax.numpy.finfo(dtype).eps
  x, _, _, _ = gmres["gmres_m"](A_mv, [], b, x0, tol, tol, n_kry, maxiter)
  solution = jax.numpy.array([2., 1.], dtype=dtype)
  np.testing.assert_allclose(x, solution, atol=tol)


@pytest.mark.parametrize("dtype", jax_dtypes)
def test_gmres_krylov(dtype):
  """
  gmres_krylov correctly builds the QR-decomposed Arnoldi decomposition.
  This function assumes that gmres["kth_arnoldi_step (which is
  independently tested) is correct.
  """
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype
  gmres = jitted_functions.gmres_wrapper(jax)

  n = 2
  n_kry = n
  np.random.seed(10)
  @jax.tree_util.Partial
  def A_mv(x):
    return A @ x
  A = jax.numpy.array(np.random.rand(n, n).astype(dtype))
  tol = A.size*jax.numpy.finfo(dtype).eps
  x0 = jax.numpy.array(np.random.rand(n).astype(dtype))
  b = jax.numpy.array(np.random.rand(n), dtype=dtype)
  r, beta = gmres["gmres_residual"](A_mv, [], b, x0)
  _, V, R, _ = gmres["gmres_krylov"](A_mv, [], n_kry, x0, r, beta,
                                     tol, jax.numpy.linalg.norm(b))
  phases = jax.numpy.sign(jax.numpy.diagonal(R[:-1, :]))
  R = phases.conj()[:, None] * R[:-1, :]
  Vtest = np.zeros((n, n_kry + 1), dtype=x0.dtype)
  Vtest[:, 0] = r/beta
  Vtest = jax.numpy.array(Vtest)
  Htest = jax.numpy.zeros((n_kry + 1, n_kry), dtype=x0.dtype)
  for k in range(n_kry):
    Vtest, Htest = gmres["kth_arnoldi_step"](k, A_mv, [], Vtest, Htest, tol)
  _, Rtest = jax.numpy.linalg.qr(Htest)
  phases = jax.numpy.sign(jax.numpy.diagonal(Rtest))
  Rtest = phases.conj()[:, None] * Rtest
  np.testing.assert_allclose(V, Vtest, atol=tol)
  np.testing.assert_allclose(R, Rtest, atol=tol)


@pytest.mark.parametrize("dtype", jax_dtypes)
def test_gs(dtype):
  """
  The Gram-Schmidt process works.
  """
  gmres = jitted_functions.gmres_wrapper(jax)
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype
  n = 8
  A = np.zeros((n, 2), dtype=dtype)
  A[:-1, 0] = 1.0
  Ai = A[:, 0] / np.linalg.norm(A[:, 0])
  A[:, 0] = Ai
  A[-1, -1] = 1.0
  A = jax.numpy.array(A)

  x0 = jax.numpy.array(np.random.rand(n).astype(dtype))
  v_new, _ = jax.lax.scan(gmres["_gs_step"], x0, xs=A.T)
  dotcheck = v_new @ A
  tol = A.size*jax.numpy.finfo(dtype).eps
  np.testing.assert_allclose(dotcheck, np.zeros(2), atol=tol)


@pytest.mark.parametrize("dtype", jax_dtypes)
def test_gmres_arnoldi_step(dtype):
  """
  The Arnoldi decomposition within GMRES is correct.
  """
  gmres = jitted_functions.gmres_wrapper(jax)
  dummy = jax.numpy.zeros(1, dtype=dtype)
  dtype = dummy.dtype
  n = 4
  n_kry = n
  np.random.seed(10)
  A = jax.numpy.array(np.random.rand(n, n).astype(dtype))
  x0 = jax.numpy.array(np.random.rand(n).astype(dtype))
  Q = np.zeros((n, n_kry + 1), dtype=x0.dtype)
  Q[:, 0] = x0/jax.numpy.linalg.norm(x0)
  Q = jax.numpy.array(Q)
  H = jax.numpy.zeros((n_kry + 1, n_kry), dtype=x0.dtype)
  tol = A.size*jax.numpy.finfo(dtype).eps
  @jax.tree_util.Partial
  def A_mv(x):
    return A @ x
  for k in range(n_kry):
    Q, H = gmres["kth_arnoldi_step"](k, A_mv, [], Q, H, tol)
  QAQ = Q[:, :n_kry].conj().T @ A @ Q[:, :n_kry]
  np.testing.assert_allclose(H[:n_kry, :], QAQ, atol=tol)


@pytest.mark.parametrize("dtype", jax_dtypes)
def test_givens(dtype):
  """
  gmres["givens_rotation produces the correct rotation factors.
  """
  gmres = jitted_functions.gmres_wrapper(jax)
  np.random.seed(10)
  v = jax.numpy.array(np.random.rand(2).astype(dtype))
  cs, sn = gmres["givens_rotation"](*v)
  rot = np.zeros((2, 2), dtype=dtype)
  rot[0, 0] = cs
  rot[1, 1] = cs
  rot[0, 1] = -sn
  rot[1, 0] = sn
  rot = jax.numpy.array(rot)
  result = rot @ v
  tol = 4*jax.numpy.finfo(dtype).eps
  np.testing.assert_allclose(result[-1], 0., atol=tol)
