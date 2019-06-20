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
"""Greedy Contraction Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pytest
from typing import List, Optional, Tuple
from tensornetwork.contractors import greedy_contractor
from tensornetwork import network


def test_greedy_sanity_check(backend):
  net = network.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2, 2, 2)))
  b = net.add_node(np.ones((2, 2, 2)))
  c = net.add_node(np.ones((2, 2, 2)))
  net.connect(a[0], a[1])
  net.connect(a[2], b[0])
  net.connect(a[3], c[0])
  net.connect(b[1], c[1])
  net.connect(b[2], c[2])
  node = greedy_contractor.greedy(net).get_final_node()
  np.testing.assert_allclose(node.tensor, np.ones(2) * 32.0)