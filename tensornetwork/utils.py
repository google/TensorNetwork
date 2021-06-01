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

import h5py
import numpy as np

from tensornetwork.component_factory import get_component
import tensornetwork.network_components as network_components
from tensornetwork.network_components import Edge, AbstractNode, Node
from tensornetwork.network_operations import reachable, get_all_edges
from typing import List, Union, BinaryIO

STRING_ENCODING = network_components.STRING_ENCODING
string_type = network_components.string_type


def save_nodes(nodes: List[AbstractNode], path: Union[str, BinaryIO]) -> None:
  """Save an iterable of nodes into hdf5 format.

  Args:
    nodes: An iterable of connected nodes. All nodes have to connect within
      `nodes`.
    path: path to file where network is saved.
  """
  if reachable(nodes) > set(nodes):
    raise ValueError(
        "Some nodes in `nodes` are connected to nodes not contained in `nodes`."
        " Saving not possible.")
  if len(set(nodes)) < len(list(nodes)):
    raise ValueError(
        'Some nodes in `nodes` appear more than once. This is not supported')
  #we need to iterate twice and order matters
  edges = list(get_all_edges(nodes))
  nodes = list(nodes)

  old_edge_names = {n: edge.name for n, edge in enumerate(edges)}
  old_node_names = {n: node.name for n, node in enumerate(nodes)}

  #generate unique names for nodes and edges
  #for saving them
  for n, node in enumerate(nodes):
    node.set_name('node{}'.format(n))

  for e, edge in enumerate(edges):
    edge.set_name('edge{}'.format(e))

  with h5py.File(path, 'w') as net_file:
    nodes_group = net_file.create_group('nodes')
    node_names_group = net_file.create_group('node_names')
    node_names_group.create_dataset(
        'names',
        dtype=string_type,
        data=np.array(list(old_node_names.values()), dtype=object))

    edges_group = net_file.create_group('edges')
    edge_names_group = net_file.create_group('edge_names')
    edge_names_group.create_dataset(
        'names',
        dtype=string_type,
        data=np.array(list(old_edge_names.values()), dtype=object))

    for n, node in enumerate(nodes):
      node_group = nodes_group.create_group(node.name)
      node._save_node(node_group)
      for edge in node.edges:
        if edge.node1 == node and edge in edges:
          edge_group = edges_group.create_group(edge.name)
          edge._save_edge(edge_group)
          edges.remove(edge)

  #name edges and nodes back  to their original names
  for n, node in enumerate(nodes):
    nodes[n].set_name(old_node_names[n])

  for n, edge in enumerate(edges):
    edges[n].set_name(old_edge_names[n])


def load_nodes(path: str) -> List[AbstractNode]:
  """Load a set of nodes from disk.

  Args:
    path: path to file where network is saved.
  Returns:
    An iterable of `Node` objects
  """
  nodes_list = []
  edges_list = []
  with h5py.File(path, 'r') as net_file:
    nodes = list(net_file["nodes"].keys())
    node_names = {
        'node{}'.format(n): v for n, v in enumerate(
            net_file["node_names"]['names'].asstr(STRING_ENCODING)[()])#pylint: disable=no-member
    }
    edge_names = {
        'edge{}'.format(n): v for n, v in enumerate(
            net_file["edge_names"]['names'].asstr(STRING_ENCODING)[()])#pylint: disable=no-member
    }
    edges = list(net_file["edges"].keys())
    for node_name in nodes:
      node_data = net_file["nodes/" + node_name]
      node_type = get_component(node_data['type'].asstr()[()])
      nodes_list.append(node_type._load_node(node_data=node_data))
    nodes_dict = {node.name: node for node in nodes_list}
    for edge in edges:
      edge_data = net_file["edges/" + edge]
      edges_list.append(Edge._load_edge(edge_data, nodes_dict))

  for edge in edges_list:
    edge.set_name(edge_names[edge.name])
  for node in nodes_list:
    node.set_name(node_names[node.name])

  return nodes_list

def from_topology(topology, tensors, backend=None):
  """Create and connect new `tn.Node`s by the given einsum-like topology.
  
  Example:
    ```
    a, b, c = tn.from_topology("xy,yz,zx", [a, b, c])
    ```
  Args:
    topology: A string that defines the topology. Should be like
      the left side of an einsum expression.
    tensors: The tensors needed to create the nodes.

  Returns:
    A list of Nodes.
  """
  edge_dict = {}
  nodes = []
  split_list = topology.split(",")
  if len(split_list) != len(tensors):
    raise ValueError("topology and number of tensors is mismatched")
  for local_axes, tensor in zip(split_list, tensors):
    local_axes_list = list(local_axes)
    if len(local_axes_list) != len(tensor.shape):
      raise ValueError(f"{local_axes} does not match shape {tensor.shape}")
    new_node = Node(tensor, axis_names=local_axes_list, backend=backend)
    for c in local_axes:
      if c in edge_dict:
        edge_dict[c] = edge_dict[c] ^ new_node[c]
      else:
        edge_dict[c] = new_node[c]
    nodes.append(new_node)
  return nodes
