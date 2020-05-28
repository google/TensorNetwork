
# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to initialize Node using a NumPy-like syntax."""

import warnings
from typing import Optional, Sequence, Tuple, Any, Union, Type, Callable, List
from typing import Text
import numpy as np
from tensornetwork.backends import base_backend
from tensornetwork import backend_contextmanager
from tensornetwork import backends
from tensornetwork import network_components


Tensor = Any
BaseBackend = base_backend.BaseBackend

# INITIALIZATION
def eye(N: int,
        dtype: Optional[Type[np.number]],
        M: Optional[int] = None,
        name: Optional[Text] = None,
        axis_names: Optional[List[Text]] = None,
        backend: Optional[Union[Text, BaseBackend]] = None) -> Tensor:
  """Return a Node representing a 2D array with ones on the diagonal and
  zeros elsewhere. The Node has two dangling Edges. The block-sparse
  backend is not supported.
  Args:
    N (int): The first dimension of the returned matrix.
    dtype, optional: The dtype of the returned matrix.
    M (int, optional): The second dimension of the returned matrix.
    name (text, optional): Name of the Node.
    axis_names (optional): List of names of the edges.
    backend (optional): The backend or its name.

  Returns:
    I : Node of shape (N, M)
        Represents an array of all zeros except for the k'th diagonal of all
        ones.
  """
  if backend is None:
    backend_obj = backend_contextmanager.get_default_backend()
  else:
    backend_obj = backends.backend_factory.get_backend(backend)
  data = backend_obj.eye(N, dtype=dtype, M=M)
  if axis_names is not None:
    if len(axis_names) != 2:
      raise ValueError("Must provide either no or exactly 2 axis_names.")

  the_node = network_components.Node(data, name=name, axis_names=axis_names,
                                     backend=backend)
  return the_node
