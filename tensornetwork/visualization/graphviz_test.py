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
"""Test of TensorNetwork Graphviz visualization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import graphviz
from tensornetwork.visualization.graphviz import to_graphviz
import tensornetwork
import numpy as np

def test_sanity_check():
  net = tensornetwork.TensorNetwork(backend="tensorflow")
  a = net.add_node(np.eye(2))
  b = net.add_node(np.eye(2))
  net.connect(a[0], b[0])
  g = to_graphviz(net)
  assert isinstance(g, graphviz.Graph)