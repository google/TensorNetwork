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
import tensornetwork
import io
import numpy as np
import h5py


def test_save_makes_hdf5_file(tmp_path, backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2)))
  b = net.add_node(np.ones((2, 2, 2)))
  net.connect(a[0], b[0])
  p = tmp_path / "network"
  net.save(p)
  assert p.exists()


def test_save_makes_hdf5_filelike_io(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2)))
  b = net.add_node(np.ones((2, 2, 2)))
  net.connect(a[0], b[0])
  p = io.BytesIO()
  net.save(p)
  assert isinstance(p.getvalue(), bytes)


def test_save_makes_hdf5_file_with_correct_substructure(tmp_path, backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(
      np.ones((2, 2, 2)), name="node_a", axis_names=["e1", "e2", "e3"])
  b = net.add_node(
      np.ones((2, 2, 2)), name="node_b", axis_names=["f1", "f2", "f3"])
  e1 = net.connect(a[0], b[0])
  e1.name = "edge_ab"
  p = tmp_path / "network"
  net.save(p)
  with h5py.File(p, 'r') as net_file:
    assert set(list(net_file.keys())) == {"backend", "nodes", "edges"}
    assert set(list(net_file['nodes'])) == {"node_a", "node_b"}
    assert set(list(net_file['edges'])) == {"edge_ab", "e2", "e3", "f2", "f3"}
    assert set(list(net_file['nodes/'])) == {"node_a", "node_b"}
    assert set(list(net_file['nodes/node_a'])) == {
        'shape', 'signature', 'backend', 'name', 'edges', 'type', 'axis_names',
        'tensor'
    }
    assert set(list(net_file['edges/edge_ab'])) == {
        'axis1', 'axis2', 'name', 'node1', 'node2', 'signature'
    }
    assert set(list(
        net_file['edges/e2'])) == {'axis1', 'name', 'node1', 'signature'}


def test_save_and_load_returns_same_network(tmp_path, backend):
  saved_net = tensornetwork.TensorNetwork(backend=backend)
  a = saved_net.add_node(
      np.ones((2, 2, 2)), name="node_a", axis_names=["e1", "e2", "e3"])
  b = saved_net.add_node(
      2 * np.ones((2, 2, 2)), name="node_b", axis_names=["f1", "f2", "f3"])
  e1 = saved_net.connect(a[0], b[0])
  e1.name = "edge_ab"

  p = tmp_path / "network"
  saved_net.save(p)

  loaded_net = tensornetwork.load(p)
  saved_nodes = list(saved_net.nodes_set)
  loaded_nodes = list(loaded_net.nodes_set)
  assert len(loaded_nodes) == len(saved_nodes)
  assert set(node.name for node in saved_nodes) == set(
      node.name for node in loaded_nodes)

  saved_edges = saved_net.get_all_edges()
  loaded_edges = loaded_net.get_all_edges()
  assert len(loaded_edges) == len(saved_edges)
  assert set(edge.name for edge in saved_edges) == set(
      edge.name for edge in loaded_edges)

  saved_node_a = [node for node in saved_nodes if node.name == "node_a"][0]
  loaded_node_a = [node for node in saved_nodes if node.name == "node_a"][0]
  np.testing.assert_allclose(saved_node_a.tensor, loaded_node_a.tensor)
  assert saved_node_a.axis_names == loaded_node_a.axis_names
  assert saved_node_a.signature == loaded_node_a.signature
  assert saved_node_a.backend.name == loaded_node_a.backend.name

  saved_node_b = [node for node in saved_nodes if node.name == "node_b"][0]
  loaded_node_b = [node for node in saved_nodes if node.name == "node_b"][0]
  np.testing.assert_allclose(saved_node_b.tensor, loaded_node_b.tensor)
  assert saved_node_b.axis_names == loaded_node_b.axis_names
  assert saved_node_b.signature == loaded_node_b.signature
  assert saved_node_b.backend.name == loaded_node_b.backend.name

  saved_edge_ab = [edge for edge in saved_edges if edge.name == "edge_ab"][0]
  loaded_edge_ab = [edge for edge in loaded_edges if edge.name == "edge_ab"][0]
  assert saved_edge_ab.node1.name == loaded_edge_ab.node1.name
  assert saved_edge_ab.node2.name == loaded_edge_ab.node2.name
  assert saved_edge_ab.signature == loaded_edge_ab.signature

  saved_edge_e2 = [edge for edge in saved_edges if edge.name == "e2"][0]
  loaded_edge_e2 = [edge for edge in loaded_edges if edge.name == "e2"][0]
  assert saved_edge_e2.node1.name == loaded_edge_e2.node1.name
  assert saved_edge_e2.node2 == loaded_edge_e2.node2
  assert saved_edge_e2.signature == loaded_edge_e2.signature


def test_save_and_load_contract_to_same_number(tmp_path, backend):
  saved_net = tensornetwork.TensorNetwork(backend=backend)
  a = saved_net.add_node(np.ones((2, 2, 2)))
  b = saved_net.add_node(2 * np.ones((2, 2, 2)))
  saved_net.connect(a[0], b[0])
  saved_net.connect(b[1], a[1])
  saved_net.connect(a[2], b[2])
  p = tmp_path / "network"
  saved_net.save(p)
  loaded_net = tensornetwork.load(p)

  saved_net.contract_between(a, b)

  loaded_nodes = list(loaded_net.nodes_set)
  loaded_net.contract_between(loaded_nodes[0], loaded_nodes[1])
  np.testing.assert_allclose(saved_net.get_final_node().tensor,
                             loaded_net.get_final_node().tensor)
