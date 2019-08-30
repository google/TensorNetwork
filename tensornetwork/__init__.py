from __future__ import absolute_import
from tensornetwork.network import TensorNetwork
from tensornetwork.network_components import Node, Edge, CopyNode
from tensornetwork.ncon_interface import ncon, ncon_network
from tensornetwork.version import __version__
from tensornetwork.visualization.graphviz import to_graphviz
from tensornetwork import contractors
from tensornetwork import config
from tensornetwork import mps
from typing import Text, Optional, Type


def set_default_backend(backend: Text, dtype: Optional[Type] = None) -> None:
  config.default_backend = backend
  config.default_dype = dtype
