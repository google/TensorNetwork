#pylint: disable=line-too-long
from tensornetwork.network_components import Node, Edge, CopyNode, BaseNode, NodeCollection
#pylint: disable=line-too-long
from tensornetwork.network_operations import norm, conj, copy, transpose, split_node, split_node_qr, split_node_rq, split_node_full_svd, reachable, check_connected, check_correct, get_all_nodes, get_all_edges, remove_node, contract_trace_edges, get_subgraph_dangling, reduced_density, switch_backend
#pylint: disable=line-too-long
from tensornetwork.network_components import contract, contract_copy_node, contract_between, outer_product, outer_product_final_nodes, contract_parallel, flatten_edges, split_edge, get_all_nondangling, get_all_dangling, flatten_all_edges, flatten_edges_between, get_parallel_edges, get_shared_edges
from tensornetwork.backends.base_backend import BaseBackend
from tensornetwork.network_components import connect, disconnect
from tensornetwork.ncon_interface import ncon, ncon_network
from tensornetwork.version import __version__
from tensornetwork.visualization.graphviz import to_graphviz
from tensornetwork import contractors
from tensornetwork import config
from typing import Text, Optional, Type, Union
from tensornetwork.utils import load_nodes, save_nodes
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
from tensornetwork.matrixproductstates.infinite_mps import InfiniteMPS


def set_default_backend(backend: Union[Text, BaseBackend]) -> None:
  config.default_backend = backend
