from tensornetwork import TensorNetwork
from typing import Set, Dict, List, Tuple

def multi_remove(elems, indices):
  """Remove multiple indicies in a list at once."""
  return [i for j, i in enumerate(elems) if j not in indices]

def get_input_sets(net: TensorNetwork) -> List[Set[int]]:
  input_sets = []
  for node in sorted(net.nodes_set):
    input_sets.append({x.signature for x in node.edges})
  return input_sets

def get_output_set(net: TensorNetwork) -> Set[int]:
  dangling_edges = net.get_all_edges() - net.get_all_nondangling()
  return {x.signature for x in dangling_edges}

def get_size_dict(net: TensorNetwork) -> Dict[int, int]:
  return {edge.signature: edge.dimension for edge in net.get_all_edges()}

