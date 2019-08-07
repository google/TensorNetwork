import opt_einsum
from tensornetwork import network
from tensornetwork.contractors.opt_einsum_paths import utils


def optimal(net: network.TensorNetwork) -> network.TensorNetwork:
  """Optimal contraction order via opt_einsum.

  This method will find the truly optimal contraction order via 
  `opt_einsum`'s depth first search algorithm. Since this search is
  exhaustive, if your network is large (n>10), then the search may
  take longer than just contracting in a suboptimal way.
  
  Args:
    net: a TensorNetwork object.

  Returns:
    The network after full contraction.
  """
  nodes = sorted(net.nodes_set)
  input_sets = utils.get_input_sets(net)
  output_set = utils.get_output_set(net)
  size_dict = utils.get_size_dict(net)
  path = opt_einsum.paths.optimal(input_sets, output_set, size_dict)
  for a, b in path:
    new_node = nodes[a] @ nodes[b]
    nodes.append(new_node)
    nodes = utils.multi_remove(nodes, [a, b])
  return net
