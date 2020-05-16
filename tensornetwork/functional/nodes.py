from collections import defaultdict
import tensornetwork as tn
import tensornetwork.functional as functional

def merge_network_nodes(network1, network2):
  nodes = network1.nodes + network2.nodes
  axes_map = defaultdict(lambda: [])
  for axes, edges in network1.axes_map.items():
    axes_map[axes] += edges
  for axes, edges in network2.axes_map.items():
    axes_map[axes] += edges
  return NetworkNode(nodes, axes_map)

class NetworkNode:
  def __init__(self, nodes, axes_map):
    self.nodes = nodes
    self.axes_map = axes_map

  def contract(self, algo, axis_order):
    node_dict, edge_dict = tn.copy(self.nodes)
    print(self.axes_map)
    for edges in self.axes_map.values():
      if len(edges) != 2 and len(edges) != 1:
        raise AssertionError(f"Bad construction {edges}")
      if len(edges) == 2:
        edges = [edge_dict[edge] for edge in edges]
        tn.connect(*edges)
    edge_order = [
        edge_dict[self.axes_map[axis][0]] for axis in axis_order
    ]
    return algo(list(node_dict.values()), output_edge_order=edge_order) 

class FunctionalNode:
  def __init__(self, tensor, axes):
    if isinstance(tensor, NetworkNode):
      self.network_node = tensor
    else:
      if not isinstance(tensor, tn.Node):
        node = tn.Node(tensor)
      else:
        node = tensor
      self.network_node = NetworkNode(
          [node], 
          {axis: [edge] for axis, edge in zip(axes, node.edges)}
      )
    self.axes_order = axes[:]

  def __call__(self, *axes_order):
    assert set(axes_order) == set(self.axes_order)
    node = self.network_node.contract(
      tn.contractors.greedy,
      axes_order)
    return FunctionalNode(node, axes_order)

  @property
  def tensor(self):
    return self.network_node.contract(
        tn.contractors.greedy, self.axes_order).tensor

  def __matmul__(self, other):
    new_axes_order = list(set(self.axes_order) ^ set(other.axes_order))
    new_network_node = merge_network_nodes(self.network_node, other.network_node)
    return FunctionalNode(new_network_node, new_axes_order)

