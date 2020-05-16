from collections import defaultdict
import tensornetwork as tn
import tensornetwork.functional as functional

def no_duplicates(values):
  """Return a set of elements from values that only appear once."""
  elements = set()
  for i in values:
    if values.count(i) == 1:
      elements.add(i)
  return elements

def merge_lazy_networks(network1, network2):
  """Merge two LazyNetworks together into a new LazyNetwork."""
  nodes = network1.nodes + network2.nodes
  axes_map = defaultdict(lambda: [])
  for axes, edges in network1.axes_map.items():
    axes_map[axes] += edges
  for axes, edges in network2.axes_map.items():
    axes_map[axes] += edges
  return LazyNetwork(nodes, axes_map)

class LazyNetwork:
  """A lazy tensor network."""
  def __init__(self, nodes, axes_map):
    self.nodes = nodes
    self.axes_map = axes_map

  def contract(self, algo, axis_order):
    """Contract the tensor network into a node.
    Args:
      algo: a contraction algorithm.
      axis_order: Order of the output axes for this network.
    Returns:
      the tn.Node of the contracted network.
    """
    node_dict, edge_dict = tn.copy(self.nodes)
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
  """A functional Node. This Node is immutable unlike the standard tn.Node
  """
  def __init__(self, tensor, axes=None):
    """Create a FunctionalNode.

    Args:
      tensor: A tensor, tn.Node. or LazyNode object.
      axes: An optional list of axes. Used to define
        an initial axis order.
    """
    if axes is None:
      self.node = tn.Node(tensor)
      self.axes_order = None
      return
    if isinstance(tensor, LazyNetwork):
      self.lazy_network = tensor
    else:
      node = tn.Node(tensor)
      axes_map = defaultdict(lambda: [])
      for axis, edge in zip(axes, node.edges):
        axes_map[axis] += [edge]
      self.lazy_network = LazyNetwork(
          [node], 
          axes_map
      )
    self.axes_order = axes[:]

  def __call__(self, *axes_order):
    """Generate a permutation of the given FuncationNode.

    Args:
      *axes_order: The new order of the axes.
    Returns:
      a new FunctionalNode with the given order.
    """ 
    if self.axes_order is None:
      return FunctionalNode(self.node, axes_order)
    return FunctionalNode(self.lazy_network, axes_order)

  @property
  def tensor(self):
    return self.lazy_network.contract(
        tn.contractors.greedy, self.axes_order).tensor

  def __matmul__(self, other):
    left_axes = no_duplicates(self.axes_order)
    right_axes = no_duplicates(other.axes_order)
    new_axes_order = list(left_axes ^ right_axes)
    new_lazy_network = merge_lazy_networks(self.lazy_network, other.lazy_network)
    return FunctionalNode(new_lazy_network, new_axes_order)

