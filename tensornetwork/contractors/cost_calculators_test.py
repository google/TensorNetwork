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
"""Cost Calculator Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pytest
from typing import List, Optional, Tuple
from tensornetwork.contractors import cost_calculators
from tensornetwork import network


def test_cost_contract_between(backend):
  net = network.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 3, 4, 5)))
  b = net.add_node(np.ones((7, 3, 9, 5)))
  net.connect(a[1], b[1])
  net.connect(a[3], b[3])
  cost = cost_calculators.cost_contract_between(a, b)
  assert cost == 2 * 7 * 4 * 9


def test_cost_contract_between_no_shared_edges(backend):
  net = network.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 3, 4, 5)))
  b = net.add_node(np.ones((7, 3, 9, 5)))
  with pytest.raises(ValueError):
    cost_calculators.cost_contract_between(a, b)


def test_cost_contract_parallel(backend):
  net = network.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 3, 4, 5)))
  b = net.add_node(np.ones((7, 3, 9, 5)))
  net.connect(a[1], b[1])
  edge = net.connect(a[3], b[3])
  cost = cost_calculators.cost_contract_parallel(edge)
  assert cost == 2 * 7 * 4 * 9