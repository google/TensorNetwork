import tensorflow as tf
import numpy as np
import scipy as sp
import jax
import pytest
from tensornetwork.backends.jax import jax_backend
import jax.config as config
import improved_gmres as gmres
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

  def A_mv(x):
    return A @ x
  tol = 100*jax.numpy.finfo(dtype).eps
  x, _, _, _ = gmres.gmres_m(A_mv, [], b, x0, tol, tol, n_kry, maxiter)
  solution = jax.numpy.array([2., 1.], dtype=dtype)
  eps = jax.numpy.linalg.norm(jax.numpy.abs(solution) - jax.numpy.abs(x))
  assert eps < tol
