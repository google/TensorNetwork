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
"""Tests for tensornetwork.backends.shell.shell_backend."""

import numpy as np
import pytest
from tensornetwork.backends.shell import shell_backend
from tensornetwork.backends.numpy import numpy_backend


def assertBackendsAgree(f, args):
  np_result = getattr(numpy_backend.NumPyBackend(), f)(**args)
  sh_result = getattr(shell_backend.ShellBackend(), f)(**args)
  assert np_result.shape == sh_result.shape


def test_shell_tensor_reshape():
  shell_tensor = shell_backend.ShellTensor((2, 1), np.float64)
  shell_tensor = shell_tensor.reshape((1, 2))
  assert shell_tensor.shape == (1, 2)


def test_tensordot():
  args = {}
  args["a"] = np.ones([3, 5, 2])
  args["b"] = np.ones([2, 3])
  args["axes"] = [[0, 2], [1, 0]]
  assertBackendsAgree("tensordot", args)


def test_reshape():
  args = {"tensor": np.ones([3, 5, 2]), "shape": np.array([3, 10])}
  assertBackendsAgree("reshape", args)


def test_transpose():
  args = {"tensor": np.ones([3, 5, 2]), "perm": [0, 2, 1]}
  assertBackendsAgree("transpose", args)


def test_svd_decomposition():
  tensor = np.ones([2, 3, 4, 5, 6])
  np_res = numpy_backend.NumPyBackend().svd_decomposition(tensor, 3)
  sh_res = shell_backend.ShellBackend().svd_decomposition(tensor, 3)
  for x, y in zip(np_res, sh_res):
    assert x.shape == y.shape


def test_svd_decomposition_raises_error():
  tensor = np.ones([2, 3, 4, 5, 6])
  with pytest.raises(NotImplementedError):
    shell_backend.ShellBackend().svd_decomposition(
        tensor, 3, max_truncation_error=.1)


def test_gmres_not_implemented():
  backend = shell_backend.ShellBackend()
  with pytest.raises(NotImplementedError):
    backend.gmres(lambda x: x, np.ones((2)))


def test_svd_decomposition_with_max_values():
  tensor = np.ones([2, 3, 4, 5, 6])
  np_res = numpy_backend.NumPyBackend().svd_decomposition(
      tensor, 3, max_singular_values=5)
  sh_res = shell_backend.ShellBackend().svd_decomposition(
      tensor, 3, max_singular_values=5)
  for x, y in zip(np_res, sh_res):
    assert x.shape == y.shape


def test_qr_decomposition():
  tensor = np.ones([2, 3, 4, 5, 6])
  np_res = numpy_backend.NumPyBackend().qr_decomposition(tensor, 3)
  sh_res = shell_backend.ShellBackend().qr_decomposition(tensor, 3)
  for x, y in zip(np_res, sh_res):
    assert x.shape == y.shape


def test_rq_decomposition():
  tensor = np.ones([2, 3, 4, 5, 6])
  np_res = numpy_backend.NumPyBackend().rq_decomposition(tensor, 3)
  sh_res = shell_backend.ShellBackend().rq_decomposition(tensor, 3)
  for x, y in zip(np_res, sh_res):
    assert x.shape == y.shape


def test_shape_concat():
  args = {
      "values": [np.ones([3, 2, 5]),
                 np.zeros([3, 2, 5]),
                 np.ones([3, 3, 5])]
  }
  args["axis"] = 1
  assertBackendsAgree("shape_concat", args)
  args["axis"] = -2
  assertBackendsAgree("shape_concat", args)


def test_concat_shape():
  shapes = [(5, 2), (3,), (4, 6)]
  result = shell_backend.ShellBackend().concat_shape(shapes)
  assert result == (5, 2, 3, 4, 6)


def test_shape_tensor():
  tensor = np.ones([3, 5, 2])
  np_result = numpy_backend.NumPyBackend().shape_tensor(tensor)
  sh_result = shell_backend.ShellBackend().shape_tensor(tensor)
  assert np_result == sh_result


def test_shape_tuple():
  tensor = np.ones([3, 5, 2])
  np_result = numpy_backend.NumPyBackend().shape_tuple(tensor)
  sh_result = shell_backend.ShellBackend().shape_tuple(tensor)
  assert np_result == sh_result


def test_shape_prod():
  result = shell_backend.ShellBackend().shape_prod(np.ones([3, 5, 2]))
  assert result == 30


def test_sqrt():
  args = {"tensor": np.ones([3, 5, 2])}
  assertBackendsAgree("sqrt", args)


def test_diag():
  args = {"tensor": np.ones(10)}
  assertBackendsAgree("diag", args)


def test_convert_to_tensor():
  args = {"tensor": np.ones([3, 5, 2])}
  assertBackendsAgree("convert_to_tensor", args)


def test_trace():
  args = {"tensor": np.ones([3, 5, 4, 4])}
  assertBackendsAgree("trace", args)


def test_outer_product():
  args = {"tensor1": np.ones([3, 5]), "tensor2": np.ones([4, 6])}
  assertBackendsAgree("outer_product", args)


def test_einsum():
  expression = "ab,bc->ac"
  tensor1, tensor2 = np.ones([5, 3]), np.ones([3, 6])
  np_result = numpy_backend.NumPyBackend().einsum(expression, tensor1, tensor2)
  sh_result = shell_backend.ShellBackend().einsum(expression, tensor1, tensor2)
  assert np_result.shape == sh_result.shape


def test_einsum_raises_error():
  expression = "ab,bc->ad"
  tensor1, tensor2 = np.ones([5, 3]), np.ones([3, 6])
  with pytest.raises(ValueError):
    shell_backend.ShellBackend().einsum(expression, tensor1, tensor2)


def test_norm():
  args = {"tensor": np.ones([3, 5])}
  assertBackendsAgree("norm", args)


def test_eye():
  args = {"N": 10, "M": 8}
  assertBackendsAgree("eye", args)


def test_eye_without_M():
  args = {"N": 10}
  assertBackendsAgree("eye", args)


def test_zeros():
  args = {"shape": (10, 4)}
  assertBackendsAgree("zeros", args)


def test_conj():
  args = {"tensor": np.ones([3, 5])}
  assertBackendsAgree("conj", args)


def test_ones():
  args = {"shape": (10, 4)}
  assertBackendsAgree("ones", args)


def test_randn():
  args = {"shape": (10, 4)}
  assertBackendsAgree("randn", args)


def test_random_uniform():
  args = {"shape": (10, 4)}
  assertBackendsAgree("random_uniform", args)


def test_eigsh_lanczos_1():
  backend = shell_backend.ShellBackend()
  D = 16
  init = backend.randn((D,))
  eigvals, eigvecs = backend.eigsh_lanczos(
      lambda x: x, initial_state=init, numeig=3, reorthogonalize=True)
  for n, ev in enumerate(eigvals):
    assert eigvecs[n].shape == (D,)
    assert ev.shape == tuple()


def test_eigsh_lanczos_shape():
  backend = shell_backend.ShellBackend()
  D = 16

  def mv(x):
    return x

  eigvals, eigvecs = backend.eigsh_lanczos(
      mv, shape=(D,), dtype=np.float64, numeig=3, reorthogonalize=True)

  for n, ev in enumerate(eigvals):
    assert eigvecs[n].shape == (D,)
    assert ev.shape == tuple()


def test_eigsh_lanczos_init_shape():
  backend = shell_backend.ShellBackend()
  D = 16
  init = backend.randn((D,))

  def mv(x):
    return x

  eigvals, eigvecs = backend.eigsh_lanczos(
      mv, numeig=3, initial_state=init, reorthogonalize=True)
  for n, ev in enumerate(eigvals):
    assert eigvecs[n].shape == (D,)
    assert ev.shape == tuple()


def test_eigsh_lanczos_raises():
  backend = shell_backend.ShellBackend()
  with pytest.raises(ValueError):
    backend.eigsh_lanczos(lambda x: x, numeig=10, num_krylov_vecs=9)
  with pytest.raises(ValueError):
    backend.eigsh_lanczos(lambda x: x, numeig=2, reorthogonalize=False)
  with pytest.raises(ValueError):
    backend.eigsh_lanczos(lambda x: x, shape=(10,), dtype=None)
  with pytest.raises(ValueError):
    backend.eigsh_lanczos(lambda x: x, shape=None, dtype=np.float64)
  with pytest.raises(ValueError):
    backend.eigsh_lanczos(lambda x: x)
  with pytest.raises(TypeError):
    backend.eigsh_lanczos(lambda x: x, initial_state=[1, 2, 3])


@pytest.mark.parametrize("a, b", [
    pytest.param(np.ones((1, 2, 3)), np.ones((1, 2, 3))),
    pytest.param(2. * np.ones(()), np.ones((1, 2, 3))),
])
def test_multiply(a, b):
  args = {"tensor1": a, "tensor2": b}
  assertBackendsAgree("multiply", args)


def test_eigh():
  matrix = np.ones([3, 3])
  vals, vecs = shell_backend.ShellBackend().eigh(matrix)
  assert vals.shape == (3,)
  assert vecs.shape == (3, 3)


def test_eigs():
  backend = shell_backend.ShellBackend()
  init = shell_backend.ShellTensor((2,), np.float64)
  eta, v = backend.eigs(lambda x: x, initial_state=init, numeig=2)
  assert len(eta) == 2
  for n in range(len(eta)):
    assert v[n].shape == (2,)

  def mv(x):
    return x

  eta, v = backend.eigs(mv, shape=(2,), dtype=np.float64, numeig=2)
  assert len(eta) == 2
  for n in range(len(eta)):
    assert v[n].shape == (2,)


def test_eigs_initial_state_shape():
  backend = shell_backend.ShellBackend()

  def mv(x):
    return x

  eta, v = backend.eigs(mv, initial_state=backend.randn((2,)))
  assert len(eta) == 1
  for n in range(len(eta)):
    assert v[n].shape == (2,)


def test_eigs_raises():
  backend = shell_backend.ShellBackend()
  with pytest.raises(ValueError):
    backend.eigs(lambda x: x, numeig=10, num_krylov_vecs=10)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigs(lambda x: x, shape=(10,), dtype=None)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigs(lambda x: x, shape=None, dtype=np.float64)
  with pytest.raises(
      ValueError,
      match="if no `initial_state` is passed, then `shape` and"
      "`dtype` have to be provided"):
    backend.eigs(lambda x: x)
  with pytest.raises(TypeError):
    backend.eigs(lambda x: x, initial_state=[1, 2, 3])


def index_update():
  backend = shell_backend.ShellBackend()
  tensor_1 = np.ones([2, 3, 4])
  tensor_2 = backend.index_update(tensor_1, tensor_1 > 0.1, 0)
  assert tensor_1.shape == tensor_2.shape


def test_matrix_inv():
  backend = shell_backend.ShellBackend()
  matrix = backend.randn((4, 4), seed=10)
  inverse = backend.inv(matrix)
  assert inverse.shape == matrix.shape


def test_matrix_inv_raises():
  backend = shell_backend.ShellBackend()
  matrix = backend.randn((4, 4, 4), seed=10)
  with pytest.raises(ValueError):
    backend.inv(matrix)


def test_broadcast_right_multiplication():
  backend = shell_backend.ShellBackend()
  tensor1 = backend.randn((2, 4, 3))
  tensor2 = backend.randn((3,))
  out = backend.broadcast_right_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out.shape, [2, 4, 3])


def test_broadcast_right_multiplication_reverse_order():
  backend = shell_backend.ShellBackend()
  tensor1 = backend.randn((3,))
  tensor2 = backend.randn((3,))
  out = backend.broadcast_right_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out.shape, [3])


def test_broadcast_right_multiplication_raises():
  backend = shell_backend.ShellBackend()
  tensor1 = backend.randn((2, 4, 3))
  tensor2 = backend.randn((3, 3))
  with pytest.raises(ValueError):
    backend.broadcast_right_multiplication(tensor1, tensor2)


def test_broadcast_left_multiplication():
  backend = shell_backend.ShellBackend()
  tensor1 = backend.randn((3,))
  tensor2 = backend.randn((3, 4, 2))
  out = backend.broadcast_left_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out.shape, [3, 4, 2])


def test_broadcast_left_multiplication_reverse_order():
  backend = shell_backend.ShellBackend()
  tensor1 = backend.randn((3,))
  tensor2 = backend.randn((3,))
  out = backend.broadcast_left_multiplication(tensor1, tensor2)
  np.testing.assert_allclose(out.shape, [3])


def test_broadcast_left_multiplication_raises():
  backend = shell_backend.ShellBackend()
  tensor1 = backend.randn((3, 3))
  tensor2 = backend.randn((3, 4, 2))
  with pytest.raises(ValueError):
    backend.broadcast_left_multiplication(tensor1, tensor2)


def test_sparse_shape():
  backend = shell_backend.ShellBackend()
  tensor = backend.randn((2, 3, 4), seed=10)
  np.testing.assert_allclose(backend.sparse_shape(tensor), tensor.shape)


def test_addition():
  backend = shell_backend.ShellBackend()
  matrix = backend.randn((4, 4, 4), seed=10)
  with pytest.raises(NotImplementedError):
    backend.addition(matrix, matrix)


def test_subtraction():
  backend = shell_backend.ShellBackend()
  matrix = backend.randn((4, 4, 4), seed=10)
  with pytest.raises(NotImplementedError):
    backend.subtraction(matrix, matrix)


def test_divide():
  backend = shell_backend.ShellBackend()
  matrix = backend.randn((4, 4, 4), seed=10)
  with pytest.raises(NotImplementedError):
    backend.divide(matrix, matrix)


def test_index_update():
  backend = shell_backend.ShellBackend()
  matrix = backend.randn((4, 4, 4), seed=10)
  actual = backend.index_update(matrix, matrix, matrix)
  assert isinstance(actual, shell_backend.ShellTensor)
  assert actual.shape == (4, 4, 4)


def test_sum():
  np.random.seed(10)
  backend = shell_backend.ShellBackend()
  a = backend.randn((2, 3, 4), seed=10)
  actual = backend.sum(a, axis=(1, 2))
  np.testing.assert_allclose(actual.shape, [
      2,
  ])

  actual = backend.sum(a, axis=(1, 2), keepdims=True)
  np.testing.assert_allclose(actual.shape, [2, 1, 1])


def test_matmul():
  np.random.seed(10)
  backend = shell_backend.ShellBackend()
  a = backend.randn((10, 2, 3), seed=10)
  b = backend.randn((10, 3, 4), seed=10)
  actual = backend.matmul(a, b)
  np.testing.assert_allclose(actual.shape, [10, 2, 4])
