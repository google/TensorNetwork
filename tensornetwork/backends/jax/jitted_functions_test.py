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


test_dtypes = [np.float64, np.complex128, np.float32, np.complex64]
@pytest.mark.parametrize("dtype", test_dtypes)
def test_gmres_arnoldi_step(dtype):
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
  kth_arnoldi_step = jitted_functions.gmres_wrapper(jax)["kth_arnoldi_step"]
  for k in range(n_kry):
    Q, H = kth_arnoldi_step(k, A_mv, [], Q, H, tol)
  QAQ = Q[:, :n_kry].conj().T @ A @ Q[:, :n_kry]
  np.testing.assert_allclose(H[:n_kry, :], QAQ, atol=tol)


@pytest.mark.parametrize("dtype", test_dtypes)
def test_givens(dtype):
  np.random.seed(10)
  v = jax.numpy.array(np.random.rand(2).astype(dtype))
  givens_rotation = jitted_functions.gmres_wrapper(jax)["givens_rotation"]
  cs, sn = givens_rotation(*v)
  rot = np.zeros((2, 2), dtype=dtype)
  rot[0, 0] = cs
  rot[1, 1] = cs
  rot[0, 1] = -sn
  rot[1, 0] = sn
  rot = jax.numpy.array(rot)
  result = rot @ v
  tol = 4*jax.numpy.finfo(dtype).eps
  np.testing.assert_allclose(result[-1], 0., atol=tol)
