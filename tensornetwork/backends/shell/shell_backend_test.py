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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
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

def test_concat():
  args = {"values": [np.ones([3, 2, 5]), np.zeros([3, 2, 5]),
                     np.ones([3, 3, 5])]}
  args["axis"] = 1
  assertBackendsAgree("concat", args)
  args["axis"] = -2
  assertBackendsAgree("concat", args)

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
