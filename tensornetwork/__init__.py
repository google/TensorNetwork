from __future__ import absolute_import
from tensornetwork.network import TensorNetwork
from tensornetwork.network_components import Node, Edge
from tensornetwork.ncon_interface import ncon, ncon_network
from tensornetwork.version import __version__
from tensornetwork.visualization.graphviz import to_graphviz
from tensornetwork import contractors
from tensornetwork import config
from typing import Text
def set_default_backend(backend: Text) -> None:
  config.default_backend = backend
