import numpy as np
import jax
import pytest
from tensornetwork.backends.jax.jitted_functions import _arnoldi_factorization
import jax.config as config
config.update('jax_enable_x64', True)


def test_arnoldi_factorization():
  D = 20
  mat = np.random.rand(D, D)
  x = np.random.rand(D)
  dtype = np.float64

  @jax.tree_util.Partial
  @jax.jit
  def matvec(vector, matrix):
    return matrix @ vector

  arnoldi = _arnoldi_factorization(jax)
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
                                 np.zeros((it, Vm.shape[1])))
