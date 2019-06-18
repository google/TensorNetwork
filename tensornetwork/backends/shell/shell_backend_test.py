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
import pytest
from tensornetwork import tensornetwork_test


def get_shape(a):
  if isinstance(a, int) or isinstance(a, float):
    return tuple()
  else:
    return a.shape

def assertShapesEqual(a, b, rtol=1e-8):
  assert get_shape(a) == get_shape(b)

@pytest.fixture(
  name="backend", params=["shell"])
def backend_fixure(request):
    return request.param

# Override np.testing to check only shapes and not values
np.testing.assert_allclose = assertShapesEqual


funcs = set(dir(tensornetwork_test))
# Skip SVD tests because it is not implemented in shell backend
funcs.remove("test_split_node")
funcs.remove("test_split_node_mixed_order")
funcs.remove("test_split_node_full_svd")
# Reimplement parallel edge test to ignore decorator
funcs.remove("test_copy_tensor_parallel_edges")


for attr in funcs:
  if attr[:4] == "test":
    globals()[attr] = getattr(tensornetwork_test, attr)

def test_copy_tensor_parallel_edges():
  tensornetwork_test.test_copy_tensor_parallel_edges("shell")
