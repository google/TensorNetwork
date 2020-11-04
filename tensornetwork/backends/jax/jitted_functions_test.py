# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import jax
import pytest
from tensornetwork.backends.jax import jitted_functions
jax.config.update('jax_enable_x64', True)

jax_dtypes = [np.float32, np.float64, np.complex64, np.complex128]
precision = jax.lax.Precision.HIGHEST

@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("ncv", [10, 20, 30])
def test_arnoldi_factorization(dtype, ncv):
  np.random.seed(10)
  D = 20
  mat = np.random.rand(D, D).astype(dtype)
  x = np.random.rand(D).astype(dtype)

  @jax.tree_util.Partial
  @jax.jit
  def matvec(vector, matrix):
    return matrix @ vector

  arnoldi = jitted_functions._generate_arnoldi_factorization(jax)
  Vm = jax.numpy.zeros((ncv, D), dtype=dtype)
  H = jax.numpy.zeros((ncv, ncv), dtype=dtype)
  start = 0
  tol = 1E-5
  Vm, Hm, residual, norm, _, _ = arnoldi(matvec, [mat], x, Vm, H, start, ncv,
                                         tol, precision)
  fm = residual * norm
  em = np.zeros((1, Vm.shape[0]))
  em[0, -1] = 1
  #test arnoldi relation
  np.testing.assert_almost_equal(mat @ Vm.T - Vm.T @ Hm - fm[:, None] * em,
                                 np.zeros((D, ncv)).astype(dtype))

@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_LR_sort(dtype):
  np.random.seed(10)
  x = np.random.rand(20).astype(dtype)
  p = 10
  LR_sort = jitted_functions._LR_sort(jax)
  actual_x, actual_inds = LR_sort(p, jax.numpy.array(np.real(x)))
  exp_inds = np.argsort(x)[::-1]
  exp_x = x[exp_inds][-p:]
  np.testing.assert_allclose(exp_x, actual_x)
  np.testing.assert_allclose(exp_inds, actual_inds)

@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_SA_sort(dtype):
  np.random.seed(10)
  x = np.random.rand(20).astype(dtype)
  p = 10
  SA_sort = jitted_functions._SA_sort(jax)
  actual_x, actual_inds = SA_sort(p, jax.numpy.array(np.real(x)))
  exp_inds = np.argsort(x)
  exp_x = x[exp_inds][-p:]
  np.testing.assert_allclose(exp_x, actual_x)
  np.testing.assert_allclose(exp_inds, actual_inds)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_shifted_QR(dtype):
  np.random.seed(10)
  D = 20
  ncv = 10
  numeig = 4
  mat = np.random.rand(D, D).astype(dtype)
  Ham = mat + mat.T.conj()
  x = np.random.rand(D).astype(dtype)

  @jax.tree_util.Partial
  @jax.jit
  def matvec(vector, matrix):
    return matrix @ vector

  lanczos = jitted_functions._generate_lanczos_factorization(jax)
  shifted_QR = jitted_functions._shifted_QR(jax)
  SA_sort = jitted_functions._SA_sort(jax)

  Vm = jax.numpy.zeros((ncv, D), dtype=dtype)
  alphas = jax.numpy.zeros(ncv, dtype=dtype)
  betas = jax.numpy.zeros(ncv - 1, dtype=dtype)
  start = 0
  tol = 1E-5
  Vm, alphas, betas, residual, norm, _, _ = lanczos(matvec, [Ham], x, Vm,
                                                    alphas, betas, start, ncv,
                                                    tol, precision)

  Hm = jax.numpy.diag(alphas) + jax.numpy.diag(betas, -1) + jax.numpy.diag(
      betas.conj(), 1)
  fm = residual * norm
  em = np.zeros((1, ncv))
  em[0, -1] = 1
  #test arnoldi relation
  np.testing.assert_almost_equal(Ham @ Vm.T - Vm.T @ Hm - fm[:, None] * em,
                                 np.zeros((D, ncv)).astype(dtype))

  evals, _ = jax.numpy.linalg.eigh(Hm)
  shifts, _ = SA_sort(numeig, evals)
  Vk, Hk, fk = shifted_QR(Vm, Hm, fm, shifts, numeig)

  Vk = Vk.at[numeig:, :].set(0)
  Hk = Hk.at[numeig:, :].set(0)
  Hk = Hk.at[:, numeig:].set(0)
  ek = np.zeros((1, ncv))
  ek[0, numeig - 1] = 1.0

  np.testing.assert_almost_equal(Ham @ Vk.T - Vk.T @ Hk - fk[:, None] * ek,
                                 np.zeros((D, ncv)).astype(dtype))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("ncv", [10, 20, 30])
def test_lanczos_factorization(dtype, ncv):
  np.random.seed(10)
  D = 20
  mat = np.random.rand(D, D).astype(dtype)
  Ham = mat + mat.T.conj()
  x = np.random.rand(D).astype(dtype)

  @jax.tree_util.Partial
  @jax.jit
  def matvec(vector, matrix):
    return matrix @ vector

  lanczos = jitted_functions._generate_lanczos_factorization(jax)

  Vm = jax.numpy.zeros((ncv, D), dtype=dtype)
  alphas = jax.numpy.zeros(ncv, dtype=dtype)
  betas = jax.numpy.zeros(ncv-1, dtype=dtype)
  start = 0
  tol = 1E-5
  Vm, alphas, betas, residual, norm, _, _ = lanczos(matvec, [Ham], x, Vm,
                                                    alphas, betas, start, ncv,
                                                    tol, precision)
  Hm = jax.numpy.diag(alphas) + jax.numpy.diag(betas, -1) + jax.numpy.diag(
      betas.conj(), 1)
  fm = residual * norm
  em = np.zeros((1, Vm.shape[0]))
  em[0, -1] = 1
  #test arnoldi relation
  np.testing.assert_almost_equal(Ham @ Vm.T - Vm.T @ Hm - fm[:, None] * em,
                                 np.zeros((D, ncv)).astype(dtype))


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
  x, _, _, _ = gmres.gmres_m(A_mv, [], b, x0, tol, tol, n_kry, maxiter,
                             precision)
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
  r, beta = gmres.gmres_residual(A_mv, [], b, x0)
  _, V, R, _ = gmres.gmres_krylov(A_mv, [], n_kry, x0, r, beta,
                                  tol, jax.numpy.linalg.norm(b),
                                  precision)
  phases = jax.numpy.sign(jax.numpy.diagonal(R[:-1, :]))
  R = phases.conj()[:, None] * R[:-1, :]
  Vtest = np.zeros((n, n_kry + 1), dtype=x0.dtype)
  Vtest[:, 0] = r/beta
  Vtest = jax.numpy.array(Vtest)
  Htest = jax.numpy.zeros((n_kry + 1, n_kry), dtype=x0.dtype)
  for k in range(n_kry):
    Vtest, Htest = gmres.kth_arnoldi_step(k, A_mv, [], Vtest, Htest, tol,
                                          precision)
  _, Rtest = jax.numpy.linalg.qr(Htest)
  phases = jax.numpy.sign(jax.numpy.diagonal(Rtest))
  Rtest = phases.conj()[:, None] * Rtest
  np.testing.assert_allclose(V, Vtest, atol=tol)
  np.testing.assert_allclose(R, Rtest, atol=tol)


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
    Q, H = gmres.kth_arnoldi_step(k, A_mv, [], Q, H, tol, precision)
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
  cs, sn = gmres.givens_rotation(*v)
  rot = np.zeros((2, 2), dtype=dtype)
  rot[0, 0] = cs
  rot[1, 1] = cs
  rot[0, 1] = -sn
  rot[1, 0] = sn
  rot = jax.numpy.array(rot)
  result = rot @ v
  tol = 4*jax.numpy.finfo(dtype).eps
  np.testing.assert_allclose(result[-1], 0., atol=tol)
