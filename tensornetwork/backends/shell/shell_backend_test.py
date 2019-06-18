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

import inspect
import numpy as np
import pytest
from tensornetwork import tensornetwork_test

shell_tests = list(dir(tensornetwork_test))
# Skip SVD tests because it is not implemented in shell backend
shell_tests.remove("test_split_node")
shell_tests.remove("test_split_node_mixed_order")
shell_tests.remove("test_split_node_full_svd")
shell_tests.remove("test_copy_tensor_parallel_edges")

def get_shape(a):
  if isinstance(a, (int, float)):
    return tuple()
  return a.shape

def assertShapesEqual(a, b, rtol=1e-8, atol=1e-8):
  assert get_shape(a) == get_shape(b)

@pytest.fixture(
  name="backend", params=["shell"])
def backend_fixure(request):
    return request.param

# Override np.testing to test only shapes
np.testing.assert_allclose = assertShapesEqual

# Reimplement copy tensor parallel edge test to ignore decorator
def test_copy_tensor_parallel_edges(backend):
  tensornetwork_test.test_copy_tensor_parallel_edges(backend)

for attr in shell_tests:
  if attr[:4] == "test":
    globals()[attr] = getattr(tensornetwork_test, attr)
