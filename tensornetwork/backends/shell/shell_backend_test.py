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
from tensornetwork.tensornetwork_test import *


def get_shape(a):
  if isinstance(a, int) or isinstance(a, float):
    return tuple()
  else:
    return a.shape

def assertShapesEqual(a, b, rtol=1e-8):
  assert get_shape(a) == get_shape(b)

# Override np.testing to check only shapes and not values
np.testing.assert_allclose = assertShapesEqual


@pytest.fixture(
  name="backend", params=["shell"])
def backend_fixure(request):
    return request.param

# Disable SVD tests since this is not implemented in shell backend
def test_split_node(backend):
  pass

def test_split_node_mixed_order(backend):
  pass

def test_split_node_full_svd(backend):
  pass

# Redefine this specific test to get rid of decorator from `tensornetwork_test`
cache_copy_tensor_parallel_edges = test_copy_tensor_parallel_edges
def test_copy_tensor_parallel_edges(backend):
  cache_copy_tensor_parallel_edges(backend)
