from tensornetwork import network
from tensornetwork import network_components
from typing import Dict, Tuple, Sequence

def reduced_density_matrix(
    network: network.TensorNetwork, 
    traced_out_edges: Sequence[network_components.Edge]
    ) -> Tuple[ 
        Dict[network_components.BaseNode, network_components.BaseNode],
        Dict[network_components.Edge, network_components.Edge]
    ]:
  """Turn a `TensorNetwork` into the reduced density matrix.

  This will modify the given `TensorNetwork` object in place to created
  a new network that represents a reduced density matrix with the given
  edges traced out.

  Args:
    network: The given `TensorNetwork`.
    traced_out_edges: `Edges` that you wished to be traced out.
  
  Returns:
    A tuple of:
      nodes_dict: A dictionary of the original `Node`s to the conjugated-copied
        `Node`s.
      edges_dict: A dictionary of the original `Edge`s to the conjugated-copied
        `Edges`s.
  """
  conj_net, nodes_dict, edges_dict = network.copy(conj=True)
  network.add_subnetwork(conj_net)
  for edge in traced_out_edges:
    edges_dict[edge] = edge ^ edges_dict[edge]
  return nodes_dict, edges_dict
