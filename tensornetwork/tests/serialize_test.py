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

import tensornetwork as tn
import pytest
import numpy as np


def assert_nodes_eq(a, b):
  assert type(a) == type(b)  #pylint: disable=unidiomatic-typecheck
  assert getattr(a, 'name', None) == getattr(b, 'name', None)
  assert getattr(a, 'axis_names', None) == getattr(b, 'axis_names', None)
  assert getattr(a, 'backend', None) == getattr(b, 'backend', None)
  assert getattr(a, 'shape', None) == getattr(b, 'shape', None)
  assert getattr(a, 'rank', None) == getattr(b, 'rank', None)
  assert getattr(a, 'dtype', None) == getattr(b, 'dtype', None)
  assert getattr(a, 'dimension', None) == getattr(b, 'dimension', None)
  ta = getattr(a, 'tensor', None)
  if isinstance(ta, np.ndarray):
    assert (ta == getattr(b, 'tensor', None)).all()


def assert_edges_eq(a, b):
  assert isinstance(a, tn.Edge) and isinstance(b, tn.Edge)
  assert a.name == b.name
  assert a._axes == b._axes


def assert_graphs_eq(a_nodes, b_nodes):
  assert len(a_nodes) == len(b_nodes)
  a_nodes_dict = {}
  b_nodes_dict = {}
  for i, (a, b) in enumerate(zip(a_nodes, b_nodes)):
    a_nodes_dict[a] = i
    b_nodes_dict[b] = i
  for a, b in zip(a_nodes, b_nodes):
    for e1, e2 in zip(a.edges, b.edges):
      assert_edges_eq(e1, e2)
      assert a_nodes_dict.get(e1.node2,
                              None) == b_nodes_dict.get(e2.node2, None)


def create_basic_network():
  np.random.seed(10)
  a = tn.Node(np.random.normal(size=[8]), name='an', axis_names=['a1'])
  b = tn.Node(np.random.normal(size=[8, 8, 8]),
              name='bn',
              axis_names=['b1', 'b2', 'b3'])
  c = tn.Node(np.random.normal(size=[8, 8, 8]),
              name='cn',
              axis_names=['c1', 'c2', 'c3'])
  d = tn.Node(np.random.normal(size=[8, 8, 8]),
              name='dn',
              axis_names=['d1', 'd2', 'd3'])

  a[0] ^ b[0]
  b[1] ^ c[0]
  c[1] ^ d[0]
  c[2] ^ b[2]

  return [a, b, c, d]


def test_basic_serial():
  nodes = create_basic_network()

  s = tn.nodes_to_json(nodes)
  new_nodes, _ = tn.nodes_from_json(s)
  for x, y in zip(nodes, new_nodes):
    assert_nodes_eq(x, y)
  assert_graphs_eq(nodes, new_nodes)
  c = tn.contractors.greedy(nodes, ignore_edge_order=True)
  new_c = tn.contractors.greedy(new_nodes, ignore_edge_order=True)
  np.testing.assert_allclose(c.tensor, new_c.tensor)


def test_exlcuded_node_serial():
  nodes = create_basic_network()

  s = tn.nodes_to_json(nodes[:-1])
  new_nodes, _ = tn.nodes_from_json(s)
  for x, y in zip(nodes, new_nodes):
    assert_nodes_eq(x, y)
  with pytest.raises(AssertionError):
    assert_graphs_eq(nodes, new_nodes)
  sub_graph = nodes[:-1]
  sub_graph[-1][1].disconnect(sub_graph[-1][1].name)
  assert_graphs_eq(sub_graph, new_nodes)


def test_serial_with_bindings():
  a, b, c, d = create_basic_network()
  bindings = {}
  a[0].name = 'ea0'
  bindings['ea'] = a[0]
  for s, n in zip(['eb', 'ec', 'ed'], [b, c, d]):
    for i, e in enumerate(n.edges):
      e.name = s + str(i)
      bindings[s] = bindings.get(s, ()) + (e,)
  s = tn.nodes_to_json([a, b, c, d], edge_binding=bindings)
  _, new_bindings = tn.nodes_from_json(s)
  assert len(new_bindings) == len(bindings)
  assert bindings['ea'].name == new_bindings['ea'][0].name
  for k in ['eb', 'ec', 'ed']:
    new_names = {e.name for e in new_bindings[k]}
    names = {e.name for e in bindings[k]}
    assert names == new_names


def test_serial_non_str_keys():
  a, b, c, d = create_basic_network()
  bindings = {}
  bindings[1] = a[0]
  with pytest.raises(TypeError):
    _ = tn.nodes_to_json([a, b, c, d], edge_binding=bindings)


def test_serial_non_edge_values():
  a, b, c, d = create_basic_network()
  bindings = {}
  bindings['non_edge'] = a
  with pytest.raises(TypeError):
    _ = tn.nodes_to_json([a, b, c, d], edge_binding=bindings)


def test_serial_exclude_non_network_edges():
  a, b, c, d = create_basic_network() # pylint: disable=unused-variable
  bindings = {'include': a[0], 'boundary': b[1], 'exclude': d[0]}
  s = tn.nodes_to_json([a, b], edge_binding=bindings)
  nodes, new_bindings = tn.nodes_from_json(s)
  assert len(nodes) == 2
  assert 'include' in new_bindings and 'boundary' in new_bindings
  assert 'exclude' not in new_bindings
