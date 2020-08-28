import numpy as np
from jax import config
import pytest
import tensornetwork
import tensornetwork.linalg.krylov
import tensornetwork.linalg.initialization
from tensornetwork.tests import testing_utils

# pylint: disable=no-member
config.update("jax_enable_x64", True)
sparse_backends = ["numpy", "jax"]


@pytest.mark.parametrize("sparse_backend", sparse_backends + ["pytorch", ])
@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_matvec_cache(sparse_backend, dtype):
  shape = (4, 4)
  dtype = testing_utils.np_dtype_to_backend(sparse_backend, dtype)
  A = tensornetwork.linalg.initialization.ones(shape,
                                               backend=sparse_backend,
                                               dtype=dtype)
  B = -1.3 * tensornetwork.linalg.initialization.ones(shape,
                                                      backend=sparse_backend,
                                                      dtype=dtype)

  def matvec(B, A):
    return A @ B

  def matvec_array(B, A):
    return A.array @ B.array

  matvec_cache = tensornetwork.linalg.krylov.MatvecCache()
  mv = matvec_cache.retrieve(sparse_backend, matvec)
  result = mv(B.array, A.array)
  test = matvec_array(B, A)
  assert np.all(np.isfinite(np.ravel(result)))
  np.testing.assert_allclose(result, test)


def test_eigsh_lanczos_raises():
  n = 2
  shape = (n, n)
  tensor = tensornetwork.linalg.initialization.ones(shape,
                                                    backend="jax",
                                                    dtype=np.float64)

  def matvec(B):
    return tensor @ B
  with pytest.raises(ValueError):
    _ = tensornetwork.linalg.krylov.eigsh_lanczos(matvec,
                                                  backend=None,
                                                  x0=None)
  with pytest.raises(ValueError):
    _ = tensornetwork.linalg.krylov.eigsh_lanczos(matvec,
                                                  backend="numpy",
                                                  x0=tensor)
  with pytest.raises(TypeError):
    _ = tensornetwork.linalg.krylov.eigsh_lanczos(matvec,
                                                  backend="numpy",
                                                  x0=tensor.array)
  with pytest.raises(TypeError):
    _ = tensornetwork.linalg.krylov.eigsh_lanczos(matvec,
                                                  backend="jax",
                                                  x0=tensor,
                                                  args=[tensor.array, ])


@pytest.mark.parametrize("sparse_backend", sparse_backends + ["pytorch", ])
@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_eigsh_lanczos(sparse_backend, dtype):
  """
  Compares linalg.krylov.eigsh_lanczos with backend.eigsh_lanczos.
  """
  n = 2
  shape = (n, n)
  dtype = testing_utils.np_dtype_to_backend(sparse_backend, dtype)
  A = tensornetwork.linalg.initialization.ones(shape,
                                               backend=sparse_backend,
                                               dtype=dtype)
  x0 = tensornetwork.linalg.initialization.ones((n, 1), backend=sparse_backend,
                                                dtype=dtype)

  def matvec(B):
    return A @ B
  result = tensornetwork.linalg.krylov.eigsh_lanczos(matvec, backend=A.backend,
                                                     x0=x0, num_krylov_vecs=n-1)

  def array_matvec(B):
    return A.array @ B
  rev, reV = result
  test_result = A.backend.eigsh_lanczos(array_matvec, initial_state=x0.array,
                                        num_krylov_vecs=n-1)
  tev, teV = test_result
  assert np.all(np.isfinite(np.ravel(np.array(rev))))
  np.testing.assert_allclose(np.array(rev), np.array(tev))

  for r, t in zip(reV, teV):
    assert np.all(np.isfinite(np.ravel(r.array)))
    np.testing.assert_allclose(r.array, t)


@pytest.mark.parametrize("sparse_backend", sparse_backends + ["pytorch", ])
@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_eigsh_lanczos_with_args(sparse_backend, dtype):
  """
  Compares linalg.krylov.eigsh_lanczos with backend.eigsh_lanczos.
  """
  n = 2
  shape = (n, n)
  dtype = testing_utils.np_dtype_to_backend(sparse_backend, dtype)
  A = tensornetwork.linalg.initialization.ones(shape,
                                               backend=sparse_backend,
                                               dtype=dtype)
  x0 = tensornetwork.linalg.initialization.ones((n, 1), backend=sparse_backend,
                                                dtype=dtype)

  def matvec(B):
    return A @ B

  def matvec_args(B, A):
    return A @ B

  result = tensornetwork.linalg.krylov.eigsh_lanczos(matvec, backend=A.backend,
                                                     x0=x0, num_krylov_vecs=n-1)
  test = tensornetwork.linalg.krylov.eigsh_lanczos(matvec_args,
                                                   backend=A.backend,
                                                   x0=x0, num_krylov_vecs=n-1,
                                                   args=[A, ])
  rev, reV = result
  tev, teV = test
  assert np.all(np.isfinite(np.ravel(np.array(rev))))
  np.testing.assert_allclose(np.array(rev), np.array(tev))

  for r, t in zip(reV, teV):
    assert np.all(np.isfinite(np.ravel(r.array)))
    np.testing.assert_allclose(r.array, t.array)


def test_eigs_raises():
  n = 2
  shape = (n, n)
  tensor = tensornetwork.linalg.initialization.ones(shape,
                                                    backend="jax",
                                                    dtype=np.float64)

  def matvec(B):
    return tensor @ B
  with pytest.raises(ValueError):
    _ = tensornetwork.linalg.krylov.eigs(matvec,
                                         backend=None,
                                         x0=None)
  with pytest.raises(ValueError):
    _ = tensornetwork.linalg.krylov.eigs(matvec,
                                         backend="numpy",
                                         x0=tensor)
  with pytest.raises(TypeError):
    _ = tensornetwork.linalg.krylov.eigs(matvec,
                                         backend="numpy",
                                         x0=tensor.array)
  with pytest.raises(TypeError):
    _ = tensornetwork.linalg.krylov.eigs(matvec,
                                         backend="jax",
                                         x0=tensor,
                                         args=[tensor.array, ])


@pytest.mark.parametrize("sparse_backend", sparse_backends)
@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_eigs(sparse_backend, dtype):
  shape = (4, 4)
  tensor = tensornetwork.linalg.initialization.ones(shape,
                                                    backend=sparse_backend,
                                                    dtype=dtype)
  x0 = tensornetwork.linalg.initialization.ones(shape, backend=sparse_backend,
                                                dtype=dtype)

  def matvec(B):
    return tensor @ B

  def test_matvec(B):
    return tensor.array @ B

  result = tensornetwork.linalg.krylov.eigs(matvec, backend=sparse_backend,
                                            x0=x0, num_krylov_vecs=3,
                                            numeig=1)
  rev, reV = result
  test_result = tensor.backend.eigs(test_matvec, initial_state=x0.array,
                                    num_krylov_vecs=3, numeig=1)
  tev, _ = test_result

  for r, t, R in zip(rev, tev, reV):
    np.testing.assert_allclose(np.array(r), np.array(t))
    testR = matvec(R) / r
    np.testing.assert_allclose(testR.array, R.array, rtol=1E-5)


@pytest.mark.parametrize("sparse_backend", sparse_backends)
@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_eigs_with_args(sparse_backend, dtype):
  shape = (4, 4)
  tensor = tensornetwork.linalg.initialization.ones(shape,
                                                    backend=sparse_backend,
                                                    dtype=dtype)
  x0 = tensornetwork.linalg.initialization.ones(shape, backend=sparse_backend,
                                                dtype=dtype)

  def matvec(B):
    return tensor @ B

  def test_matvec(B, A):
    return A @ B

  result = tensornetwork.linalg.krylov.eigs(matvec, backend=sparse_backend,
                                            x0=x0, num_krylov_vecs=3,
                                            numeig=1)
  rev, _ = result
  test_result = tensornetwork.linalg.krylov.eigs(test_matvec, x0=x0,
                                                 num_krylov_vecs=3, numeig=1,
                                                 args=[tensor, ])
  tev, teV = test_result

  for r, t, R in zip(rev, tev, teV):
    np.testing.assert_allclose(np.array(r), np.array(t))
    testR = matvec(R) / r
    np.testing.assert_allclose(testR.array, R.array, rtol=1E-5)


def test_gmres_raises():
  n = 2
  shape = (n, 1)
  tensor = tensornetwork.linalg.initialization.ones(shape,
                                                    backend="jax",
                                                    dtype=np.float64)
  tensornp = tensornetwork.linalg.initialization.ones(shape,
                                                      backend="numpy",
                                                      dtype=np.float64)

  def matvec(B):
    return tensor @ B
  with pytest.raises(ValueError):
    _ = tensornetwork.linalg.krylov.gmres(matvec, tensor, x0=tensornp)
  with pytest.raises(ValueError):
    _ = tensornetwork.linalg.krylov.gmres(matvec, tensornp, x0=tensor)
  with pytest.raises(TypeError):
    _ = tensornetwork.linalg.krylov.gmres(matvec, tensor.array)
  with pytest.raises(TypeError):
    _ = tensornetwork.linalg.krylov.gmres(matvec, tensor, x0=tensor.array)
  with pytest.raises(TypeError):
    _ = tensornetwork.linalg.krylov.gmres(matvec, tensor,
                                          A_args=[tensor.array, ])


@pytest.mark.parametrize("sparse_backend", sparse_backends)
@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_gmres(dtype, sparse_backend):
  Adat = np.array(([[1, 1], [3, -4]]), dtype=dtype)
  A = tensornetwork.tensor.Tensor(Adat, backend=sparse_backend)
  bdat = np.array([3, 2], dtype=dtype).reshape((2, 1))
  b = tensornetwork.tensor.Tensor(bdat, backend=sparse_backend)
  x0dat = np.ones((2, 1), dtype=dtype)
  x0 = tensornetwork.tensor.Tensor(x0dat, backend=sparse_backend)
  n_kry = 2

  def A_mv(y):
    return A @ y

  def A_mv_arr(y):
    return A.array @ y

  x, _ = A.backend.gmres(A_mv_arr, bdat, x0=x0dat, num_krylov_vectors=n_kry)
  xT, _ = tensornetwork.linalg.krylov.gmres(A_mv, b, x0=x0,
                                            num_krylov_vectors=n_kry)
  np.testing.assert_allclose(x, xT.array)


@pytest.mark.parametrize("sparse_backend", sparse_backends)
@pytest.mark.parametrize("dtype", testing_utils.np_float_dtypes)
def test_gmres_with_args(dtype, sparse_backend):
  Adat = np.array(([[1, 1], [3, -4]]), dtype=dtype)
  A = tensornetwork.tensor.Tensor(Adat, backend=sparse_backend)
  bdat = np.array([3, 2], dtype=dtype).reshape((2, 1))
  b = tensornetwork.tensor.Tensor(bdat, backend=sparse_backend)
  x0dat = np.ones((2, 1), dtype=dtype)
  x0 = tensornetwork.tensor.Tensor(x0dat, backend=sparse_backend)
  n_kry = 2

  def A_mv(y):
    return A @ y

  def A_mv_test(y, A):
    return A @ y

  x, _ = tensornetwork.linalg.krylov.gmres(A_mv, b, x0=x0,
                                           num_krylov_vectors=n_kry)
  xT, _ = tensornetwork.linalg.krylov.gmres(A_mv_test, b, x0=x0,
                                            num_krylov_vectors=n_kry,
                                            A_args=[A, ])
  np.testing.assert_allclose(x.array, xT.array)
