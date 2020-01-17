# pytype: skip-file
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
"""Tests for tensornetwork.backends.shell.shell_backend"""

import numpy as np
import pytest
from tensornetwork.backends.shell import shell_backend
from tensornetwork.backends.numpy import numpy_backend


def assertBackendsAgree(f, args):
  np_result = getattr(numpy_backend.NumPyBackend(), f)(**args)
  sh_result = getattr(shell_backend.ShellBackend(), f)(**args)
  assert np_result.shape == sh_result.shape


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


def test_svd_decomposition_with_max_values():
  tensor = np.ones([2, 3, 4, 5, 6])
  np_res = numpy_backend.NumPyBackend().svd_decomposition(
      tensor, 3, max_singular_values=5)
  sh_res = shell_backend.ShellBackend().svd_decomposition(
      tensor, 3, max_singular_values=5)
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


def test_concat():
  np_backend = numpy_backend.NumPyBackend()
  sh_backend = shell_backend.ShellBackend()
  scalars = [np_backend.convert_to_tensor(1.0),
             np_backend.convert_to_tensor(2.0)]
  actual = sh_backend.concat(scalars, 0).shape
  expected = np.array([1.0, 2.0])
  np.testing.assert_allclose(expected, actual)


def test_concat_shape():
  shapes = [(5, 2), (3,), (4, 6)]
  result = shell_backend.ShellBackend().concat_shape(shapes)
  assert result == (5, 2, 3, 4, 6)


def test_shape():
  tensor = np.ones([3, 5, 2])
  np_result = numpy_backend.NumPyBackend().shape(tensor)
  sh_result = shell_backend.ShellBackend().shape(tensor)
  assert np_result == sh_result


def test_shape_tuple():
  tensor = np.ones([3, 5, 2])
  np_result = numpy_backend.NumPyBackend().shape_tuple(tensor)
  sh_result = shell_backend.ShellBackend().shape_tuple(tensor)
  assert np_result == sh_result


def test_prod():
  result = shell_backend.ShellBackend().prod(np.ones([3, 5, 2]))
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


def test_norm():
  args = {"tensor": np.ones([3, 5])}
  assertBackendsAgree("norm", args)


def test_eye():
  args = {"N": 10, "M": 8}
  assertBackendsAgree("eye", args)


def test_zeros():
  args = {"shape": (10, 4)}
  assertBackendsAgree("zeros", args)


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
      lambda x: x, init, numeig=3, reorthogonalize=True)
  for n, ev in enumerate(eigvals):
    assert eigvecs[n].shape == (D,)
    assert ev.shape == tuple()


def test_eigsh_lanczos_2():
  backend = shell_backend.ShellBackend()
  D = 16

  class LinearOperator:

    def __init__(self, shape):
      self.shape = shape

    def __call__(self, x):
      return x

  mv = LinearOperator(shape=((D,), (D,)))
  eigvals, eigvecs = backend.eigsh_lanczos(mv, numeig=3, reorthogonalize=True)
  for n, ev in enumerate(eigvals):
    assert eigvecs[n].shape == (D,)
    assert ev.shape == tuple()


def test_eigsh_lanczos_raises():
  backend = shell_backend.ShellBackend()
  with pytest.raises(AttributeError):
    backend.eigsh_lanczos(lambda x: x)
  with pytest.raises(ValueError):
    backend.eigsh_lanczos(lambda x: x, numeig=10, num_krylov_vecs=9)
  with pytest.raises(ValueError):
    backend.eigsh_lanczos(lambda x: x, numeig=2, reorthogonalize=False)


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
  eta, v = backend.eigs(lambda x: x, initial_state=np.random.rand(2), numeig=2)
  assert len(eta) == 2
  for n in range(len(eta)):
    assert v[n].shape == (2,)

  class MV:

    def __init__(self, shape):
      self.shape = shape

    def __call__(self, x):
      return x

  mv = MV((2, 2))
  eta, v = backend.eigs(mv, numeig=2)
  assert len(eta) == 2
  for n in range(len(eta)):
    assert v[n].shape == (2,)


def test_eigs_raises():

  class MV:

    def __init__(self, shape):
      self.shape = shape

    def __call__(self, x):
      return x

  backend = shell_backend.ShellBackend()
  mv = MV((2, 2))
  with pytest.raises(ValueError):
    backend.eigs(mv, initial_state=np.random.rand(3))
  with pytest.raises(AttributeError):
    backend.eigs(lambda x: x)


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
