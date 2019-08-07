import opt_einsum
import tensornetwork
from tensornetwork.contractors.opt_einsum_paths import utils


def optimal(net: tensornetwork.TensorNetwork) -> tensornetwork.TensorNetwork:
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
