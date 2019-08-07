from tensornetwork import network
from typing import Set, Dict, List, Tuple

def multi_remove(elems, indices):
  """Remove multiple indicies in a list at once."""
  return [i for j, i in enumerate(elems) if j not in indices]

def get_input_sets(net: network.TensorNetwork) -> List[Set[int]]:
  input_sets = []
  for node in sorted(net.nodes_set):
    input_sets.append(set(node.edges))
  return input_sets

def get_output_set(net: network.TensorNetwork) -> Set[int]:
  dangling_edges = net.get_all_edges() - net.get_all_nondangling()
  return set(dangling_edges)

def get_size_dict(net: network.TensorNetwork) -> Dict[int, int]:
  return {edge: edge.dimension for edge in net.get_all_edges()}
