
from tensornetwork.network import TensorNetwork
from tensornetwork.network_components import Node, Edge, CopyNode, BaseNode
#pylint: disable=line-too-long
from tensornetwork.network_operations import conj, copy, transpose, split_node, split_node_qr, split_node_rq, split_node_full_svd, reachable, check_connected, check_correct, get_all_nodes, get_all_edges, remove_node

from tensornetwork.network_components import contract, contract_copy_node, contract_between, outer_product, outer_product_final_nodes, contract_parallel, flatten_edges, get_all_nondangling, get_all_dangling, flatten_all_edges, flatten_edges_between, get_parallel_edges, get_shared_edges

from tensornetwork.network_components import connect, disconnect
from tensornetwork.ncon_interface import ncon, ncon_network
from tensornetwork.version import __version__
from tensornetwork.visualization.graphviz import to_graphviz
from tensornetwork import contractors
from tensornetwork import config
from typing import Text, Optional, Type
from tensornetwork.utils import load


def set_default_backend(backend: Text, dtype: Optional[Type] = None) -> None:
  config.default_backend = backend
  config.default_dtype = dtype
