import h5py
import tensornetwork


def load(path: str):
  """Load a tensor network from disk.

  Args:
    path: path to folder where network is saved.
  """
  with h5py.File(path, 'r') as net_file:
    net = tensornetwork.TensorNetwork(backend=net_file["backend"][()])
    nodes = list(net_file["nodes"].keys())
    edges = list(net_file["edges"].keys())

    for node_name in nodes:
      node_data = net_file["nodes/" + node_name]
      node_type = (tensornetwork.component_factory
                                .get_component(node_data['type'][()]))
      node_type._load_node(net, node_data)

    nodes_dict = {node.name: node for node in net.nodes_set}

    for edge in edges:
      edge_data = net_file["edges/" + edge]
      tensornetwork.network_components.Edge._load_edge(edge_data, nodes_dict)
  return net
