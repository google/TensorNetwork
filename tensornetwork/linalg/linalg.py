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
from tensornetwork.backends import abstract_backend
#pylint: disable=line-too-long
from tensornetwork.network_components import AbstractNode, Node, outer_product_final_nodes
from tensornetwork import backend_contextmanager
from tensornetwork import backends
from tensornetwork import network_components

Tensor = Any
BaseBackend = abstract_backend.AbstractBackend


# INITIALIZATION
def initialize_node(fname: Text,
                    *fargs: Any,
                    name: Optional[Text] = None,
                    axis_names: Optional[List[Text]] = None,
                    backend: Optional[Union[Text, BaseBackend]] = None,
                    **fkwargs: Any) -> Tensor:
  """Return a Node wrapping data obtained by an initialization function
  implemented in a backend. The Node will have the same shape as the
  underlying array that function generates, with all Edges dangling.

  This function is not intended to be called directly, but doing so should
  be safe enough.
  Args:
    fname:  Name of the method of backend to call (a string).
    *fargs: Positional arguments to the initialization method.
    name: Optional name of the Node.
    axis_names: Optional names of the Node's dangling edges.
    backend: The backend or its name.
    **fkwargs: Keyword arguments to the initialization method.

  Returns:
    node: A Node wrapping data generated by
          (the_backend).fname(*fargs, **fkwargs), with one dangling edge per
          axis of data.
  """
  if backend is None:
    backend_obj = backend_contextmanager.get_default_backend()
  else:
    backend_obj = backends.backend_factory.get_backend(backend)
  func = getattr(backend_obj, fname)
  data = func(*fargs, **fkwargs)
  node = Node(data, name=name, axis_names=axis_names, backend=backend)
  return node


def eye(N: int,
        dtype: Optional[Type[np.number]] = None,
        M: Optional[int] = None,
        name: Optional[Text] = None,
        axis_names: Optional[List[Text]] = None,
        backend: Optional[Union[Text, BaseBackend]] = None) -> Tensor:
  """Return a Node representing a 2D array with ones on the diagonal and
  zeros elsewhere. The Node has two dangling Edges.
  Args:
    N (int): The first dimension of the returned matrix.
    dtype, optional: dtype of array (default np.float64).
    M (int, optional): The second dimension of the returned matrix.
    name (text, optional): Name of the Node.
    axis_names (optional): List of names of the edges.
    backend (optional): The backend or its name.

  Returns:
    I : Node of shape (N, M)
        Represents an array of all zeros except for the k'th diagonal of all
        ones.
  """
  the_node = initialize_node(
      "eye",
      N,
      name=name,
      axis_names=axis_names,
      backend=backend,
      dtype=dtype,
      M=M)
  return the_node


def zeros(shape: Sequence[int],
          dtype: Optional[Type[np.number]] = None,
          name: Optional[Text] = None,
          axis_names: Optional[List[Text]] = None,
          backend: Optional[Union[Text, BaseBackend]] = None) -> Tensor:
  """Return a Node of shape `shape` of all zeros.
  The Node has one dangling Edge per dimension.
  Args:
    shape : Shape of the array.
    dtype, optional: dtype of array (default np.float64).
    name (text, optional): Name of the Node.
    axis_names (optional): List of names of the edges.
    backend (optional): The backend or its name.
  Returns:
    the_node : Node of shape `shape`. Represents an array of all zeros.
  """
  the_node = initialize_node(
      "zeros",
      shape,
      name=name,
      axis_names=axis_names,
      backend=backend,
      dtype=dtype)
  return the_node


def ones(shape: Sequence[int],
         dtype: Optional[Type[np.number]] = None,
         name: Optional[Text] = None,
         axis_names: Optional[List[Text]] = None,
         backend: Optional[Union[Text, BaseBackend]] = None) -> Tensor:
  """Return a Node of shape `shape` of all ones.
  The Node has one dangling Edge per dimension.
  Args:
    shape : Shape of the array.
    dtype, optional: dtype of array (default np.float64).
    name (text, optional): Name of the Node.
    axis_names (optional): List of names of the edges.
    backend (optional): The backend or its name.
  Returns:
    the_node : Node of shape `shape`
        Represents an array of all ones.
  """
  the_node = initialize_node(
      "ones",
      shape,
      name=name,
      axis_names=axis_names,
      backend=backend,
      dtype=dtype)
  return the_node


def ones_like(a: np.ndarray,
         dtype: Optional[Type[np.number]] = None,
         name: Optional[Text] = None,
         axis_names: Optional[List[Text]] = None,
         backend: Optional[Union[Text, BaseBackend]] = None) -> Tensor:
  """Return a Node of all ones, of same shape as `a`.
  The Node has one dangling Edge per dimension.
  Args:
    shape : Shape of the array.
    dtype, optional: dtype of array (default np.float64).
    name (text, optional): Name of the Node.
    axis_names (optional): List of names of the edges.
    backend (optional): The backend or its name.
  Returns:
    the_node : Node of shape `shape`
        Represents an array of all ones.
  """
  the_node = initialize_node(
      "ones",
      a.shape,
      name=name,
      axis_names=axis_names,
      backend=backend,
      dtype=dtype)
  return the_node


def randn(shape: Sequence[int],
          dtype: Optional[Type[np.number]] = None,
          seed: Optional[int] = None,
          name: Optional[Text] = None,
          axis_names: Optional[List[Text]] = None,
          backend: Optional[Union[Text, BaseBackend]] = None) -> Tensor:
  """Return a Node of shape `shape` of Gaussian random floats.
  The Node has one dangling Edge per dimension.
  Args:
    shape : Shape of the array.
    dtype, optional: dtype of array (default np.float64).
    seed, optional: Seed for the RNG.
    name (text, optional): Name of the Node.
    axis_names (optional): List of names of the edges.
    backend (optional): The backend or its name.
  Returns:
    the_node : Node of shape `shape` filled with Gaussian random data.
  """
  the_node = initialize_node(
      "randn",
      shape,
      name=name,
      axis_names=axis_names,
      backend=backend,
      seed=seed,
      dtype=dtype)
  return the_node


def random_uniform(
    shape: Sequence[int],
    dtype: Optional[Type[np.number]] = None,
    seed: Optional[int] = None,
    boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
    name: Optional[Text] = None,
    axis_names: Optional[List[Text]] = None,
    backend: Optional[Union[Text, BaseBackend]] = None) -> Tensor:
  """Return a Node of shape `shape` of uniform random floats.
  The Node has one dangling Edge per dimension.
  Args:
    shape : Shape of the array.
    dtype, optional: dtype of array (default np.float64).
    seed, optional: Seed for the RNG.
    boundaries : Values lie in [boundaries[0], boundaries[1]).
    name (text, optional): Name of the Node.
    axis_names (optional): List of names of the edges.
    backend (optional): The backend or its name.
  Returns:
    the_node : Node of shape `shape` filled with uniform random data.
  """
  the_node = initialize_node(
      "random_uniform",
      shape,
      name=name,
      axis_names=axis_names,
      backend=backend,
      seed=seed,
      boundaries=boundaries,
      dtype=dtype)
  return the_node


def norm(node: AbstractNode) -> Tensor:
  """The L2 norm of `node`

  Args:
    node: A `AbstractNode`.

  Returns:
    The L2 norm.

  Raises:
    AttributeError: If `node` has no `backend` attribute.
  """
  if not hasattr(node, 'backend'):
    raise AttributeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))
  return node.backend.norm(node.tensor)


def conj(node: AbstractNode,
         name: Optional[Text] = None,
         axis_names: Optional[List[Text]] = None) -> AbstractNode:
  """Conjugate a `node`.

  Args:
    node: A `AbstractNode`.
    name: Optional name to give the new node.
    axis_names: Optional list of names for the axis.

  Returns:
    A new node. The complex conjugate of `node`.

  Raises:
    AttributeError: If `node` has no `backend` attribute.
  """
  if not hasattr(node, 'backend'):
    raise AttributeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))
  backend = node.backend
  if not axis_names:
    axis_names = node.axis_names

  return Node(
      backend.conj(node.tensor),
      name=name,
      axis_names=axis_names,
      backend=backend)


def transpose(node: AbstractNode,
              permutation: Sequence[Union[Text, int]],
              name: Optional[Text] = None,
              axis_names: Optional[List[Text]] = None) -> AbstractNode:
  """Transpose `node`

  Args:
    node: A `AbstractNode`.
    permutation: A list of int or str. The permutation of the axis.
    name: Optional name to give the new node.
    axis_names: Optional list of names for the axis.

  Returns:
    A new node. The transpose of `node`.

  Raises:
    AttributeError: If `node` has no `backend` attribute, or if
      `node` has no tensor.
    ValueError: If either `permutation` is not the same as expected or
      if you try to permute with a trace edge.
  """

  if not hasattr(node, 'backend'):
    raise AttributeError('Node {} of type {} has no `backend`'.format(
        node, type(node)))

  perm = [node.get_axis_number(p) for p in permutation]
  if not axis_names:
    axis_names = node.axis_names

  new_node = Node(
      node.tensor, name=name, axis_names=node.axis_names, backend=node.backend)
  return new_node.reorder_axes(perm)


def kron(nodes: Sequence[AbstractNode]) -> AbstractNode:
  """Kronecker product of the given nodes.

  Kronecker products of nodes is the same as the outer product, but the order
  of the axes is different. The first half of edges of all of the nodes will
  appear first half of edges in the resulting node, and the second half ot the
  edges in each node will be in the second half of the resulting node.

  For example, if I had two nodes  :math:`X_{ab}`,  :math:`Y_{cdef}`, and 
  :math:`Z_{gh}`, then the resulting node would have the edges ordered 
  :math:`R_{acdgbefh}`.
   
  The kronecker product is designed such that the kron of many operators is
  itself an operator. 

  Args:
    nodes: A sequence of `AbstractNode` objects.

  Returns:
    A `Node` that is the kronecker product of the given inputs. The first
    half of the edges of this node would represent the "input" edges of the
    operator and the last half of edges are the "output" edges of the
    operator.
  """
  input_edges = []
  output_edges = []
  for node in nodes:
    order = len(node.shape)
    if order % 2 != 0:
      raise ValueError(f"All operator tensors must have an even order. "
                       f"Found tensor with order {order}")
    input_edges += node.edges[:order // 2]
    output_edges += node.edges[order // 2:]
  result = outer_product_final_nodes(nodes, input_edges + output_edges)
  return result
