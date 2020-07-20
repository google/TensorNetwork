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

import math
import numpy as np
import torch
from tensornetwork.backends.pytorch import decompositions


def test_expected_shapes():
  val = torch.zeros((2, 3, 4, 5))
  u, s, vh, _ = decompositions.svd(torch, val, 2)
  assert u.shape == (2, 3, 6)
  assert s.shape == (6,)
  np.testing.assert_allclose(s, np.zeros(6))
  assert vh.shape == (6, 4, 5)


def test_expected_shapes_qr():
  val = torch.zeros((2, 3, 4, 5))
  for non_negative_diagonal in [True, False]:
    q, r = decompositions.qr(torch, val, 2, non_negative_diagonal)
    assert q.shape == (2, 3, 6)
    assert r.shape == (6, 4, 5)


def test_expected_shapes_rq():
  val = torch.zeros((2, 3, 4, 5))
  for non_negative_diagonal in [True, False]:
    r, q = decompositions.rq(torch, val, 2, non_negative_diagonal)
    assert r.shape == (2, 3, 6)
    assert q.shape == (6, 4, 5)


def test_rq():
  random_matrix = torch.rand([10, 10], dtype=torch.float64)
  for non_negative_diagonal in [True, False]:
    r, q = decompositions.rq(torch, random_matrix, 1, non_negative_diagonal)
    np.testing.assert_allclose(r.mm(q), random_matrix)


def test_qr():
  random_matrix = torch.rand([10, 10], dtype=torch.float64)
  for non_negative_diagonal in [True, False]:
    q, r = decompositions.rq(torch, random_matrix, 1, non_negative_diagonal)
    np.testing.assert_allclose(q.mm(r), random_matrix)


def test_max_singular_values():
  np.random.seed(2018)
  random_matrix = np.random.rand(10, 10)
  unitary1, _, unitary2 = np.linalg.svd(random_matrix)
  singular_values = np.array(range(10))
  val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
  u, s, vh, trun = decompositions.svd(
      torch, torch.tensor(val), 1, max_singular_values=7)
  assert u.shape == (10, 7)
  assert s.shape == (7,)
  np.testing.assert_array_almost_equal(s, np.arange(9, 2, -1))
  assert vh.shape == (7, 10)
  np.testing.assert_array_almost_equal(trun, np.arange(2, -1, -1))


def test_max_truncation_error():
  np.random.seed(2019)
  random_matrix = np.random.rand(10, 10)
  unitary1, _, unitary2 = np.linalg.svd(random_matrix)
  singular_values = np.array(range(10))
  val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
  u, s, vh, trun = decompositions.svd(
      torch, torch.Tensor(val), 1, max_truncation_error=math.sqrt(5.1))
  assert u.shape == (10, 7)
  assert s.shape == (7,)
  np.testing.assert_array_almost_equal(s, np.arange(9, 2, -1), decimal=5)
  assert vh.shape == (7, 10)
  np.testing.assert_array_almost_equal(trun, np.arange(2, -1, -1))


def test_max_truncation_error_relative():
  absolute = np.diag([2.0, 1.0, 0.2, 0.1])
  relative = np.diag([2.0, 1.0, 0.2, 0.1])
  max_truncation_err = 0.2
  _, _, _, trunc_sv_absolute = decompositions.svd(
      torch,
      torch.Tensor(absolute),
      1,
      max_truncation_error=max_truncation_err,
      relative=False)
  _, _, _, trunc_sv_relative = decompositions.svd(
      torch,
      torch.Tensor(relative),
      1,
      max_truncation_error=max_truncation_err,
      relative=True)
  np.testing.assert_almost_equal(trunc_sv_absolute, [0.1])
  np.testing.assert_almost_equal(trunc_sv_relative, [0.2, 0.1])
