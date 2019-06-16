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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensornetwork.backends.shell import shell_backend


def test_tensordot_matmul():
  bkd = shell_backend.ShellBackend()
  a = np.ones((6, 4, 3, 10))
  b = np.ones((6, 4, 10, 5))
  axes = [[3], [2]]
  c = bkd.tensordot(a.shape, b.shape, axes=axes)
  assert c == np.tensordot(a, b, axes=axes).shape

def test_tensordot():
  bkd = shell_backend.ShellBackend()
  a = np.ones((6, 4, 3, 10))
  b = np.ones((4, 6, 10, 5))
  axes = [[0, 1, 3], [1, 0, 2]]
  c = bkd.tensordot(a.shape, b.shape, axes=axes)
  assert c == np.tensordot(a, b, axes=axes).shape

def test_reshape():
  bkd = shell_backend.ShellBackend()
  shape = bkd.reshape((5, 6), (5, 3, 2))
  assert shape == (5, 3, 2)

def test_transpose():
  bkd = shell_backend.ShellBackend()
  shape = bkd.transpose((5, 3, 2), [0, 2, 1])
  assert shape == (5, 2, 3)

def test_shape_concat():
  bkd = shell_backend.ShellBackend()
  values = [(5, 3, 2), (4, 6), (2,)]
  shape = bkd.shape_concat(values)
  assert shape == (5, 3, 2, 4, 6, 2)

def test_shape_prod():
  bkd = shell_backend.ShellBackend()
  values = (5, 3, 2)
  shape = bkd.shape_prod(values)
  assert shape == 30

def test_trace():
  bkd = shell_backend.ShellBackend()
  shape = bkd.trace((5, 3, 6, 2, 2))
  assert shape == (5, 3, 6)

def test_outer_product():
  bkd = shell_backend.ShellBackend()
  shape = bkd.outer_product((5, 2, 3), (6, 2, 4))
  assert shape == (5, 2, 3, 6, 2, 4)

def test_einsum_batch():
  bkd = shell_backend.ShellBackend()
  a = np.ones((6, 4, 3, 10))
  b = np.ones((6, 4, 10, 5))
  expr = "abij,abjk->abik"
  c = bkd.einsum(expr, a.shape, b.shape)
  assert c == np.einsum(expr, a, b).shape

def test_einsum():
  bkd = shell_backend.ShellBackend()
  a = np.ones((6, 4, 3, 10, 8))
  b = np.ones((6, 7, 4, 10, 5))
  expr = "abicj,akbcl->kilj"
  c = bkd.einsum(expr, a.shape, b.shape)
  assert c == np.einsum(expr, a, b).shape

def test_einsum_three():
  bkd = shell_backend.ShellBackend()
  a = np.ones((6, 4, 3, 10, 8))
  b = np.ones((6, 7, 4, 10, 5))
  c = np.ones((5, 3))
  expr = "abicj,akbcl,li->jk"
  s = bkd.einsum(expr, a.shape, b.shape, c.shape)
  assert s == np.einsum(expr, a, b, c).shape
