# Copyright 2019 The TensorNetwork Authors
## Licensed under the Apache License, Version 2.0 (the "License");
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
"""NCON interface to TensorNetwork."""

import warnings
import numpy as np
# pylint: disable=line-too-long
from typing import Any, Sequence, List, Optional, Union, Text, Tuple, Dict
from tensornetwork import network_components
from tensornetwork.backend_contextmanager import get_default_backend
from tensornetwork.backends import backend_factory
from tensornetwork.backends.base_backend import BaseBackend
Tensor = Any

_CACHED_JITTED_NCONS = {}


def _map_string_to_int(network_structure):
  flat_labels = np.concatenate(network_structure)
  unique, cnts = np.unique(flat_labels, return_counts=True)
  cont_labels = unique[cnts >= 2]
  out_labels = unique[cnts == 1]
  ordered_out_labels = np.sort(out_labels)

  mapping = dict(zip(cont_labels, np.arange(1, len(cont_labels) + 1)))
  out_mapping = dict(
      zip(ordered_out_labels, -np.arange(1,
                                         len(ordered_out_labels) + 1)))
  mapping.update(out_mapping)
  mapped_network_structure = [
      np.array([
          mapping[str(label)]  #`label` we force 'label' a `str`
          for label in labels
      ])
      for labels in network_structure
  ]
  return mapped_network_structure, mapping


def _check_network(network_structure, tensor_dimensions, flat_connections,
                   con_order, out_order, reverse_mapping):
  flat_dimensions = np.concatenate(tensor_dimensions)
  pos_cons = flat_connections[flat_connections > 0]
  neg_cons = flat_connections[flat_connections < 0]

  unique, cnts = np.unique(flat_connections, return_counts=True)
  if reverse_mapping is None:
    reverse_mapping = dict(zip(unique, unique))

  pos_connections = unique[unique > 0]
  pos_cnts = cnts[unique > 0]
  neg_connections = unique[unique < 0]
  neg_cnts = cnts[unique < 0]
  if len(network_structure) != len(tensor_dimensions):
    raise ValueError("number of tensors does not match the"
                     " number of network connections.")

  for n, dims in enumerate(tensor_dimensions):
    if len(dims) != len(network_structure[n]):
      raise ValueError(f"number of indices does not match"
                       f" number of labels on tensor {n}.")
  # check `con_order`
  log_array = np.isin(con_order, pos_cons)
  if not np.all(log_array):
    raise ValueError(
        (f"labels {con_order[np.logical_not(log_array)]} in `con_order` "
         f"do not appear in network structure."))

  unique_con, con_cnts = np.unique(con_order, return_counts=True)
  if np.any(con_cnts >= 2):
    raise ValueError(
        f"contraction labels {unique_con[con_cnts>=2]} appear more than once in `con_order`."
    )

  if not np.array_equal(np.sort(con_order), np.unique(pos_cons)):
    raise ValueError((f"{[reverse_mapping[o] for o in con_order]} "
                      f"is not a valid contraction order."))

  # check `out_order`
  log_array = np.isin(out_order, neg_cons)
  if not np.all(log_array):
    raise ValueError(
        (f"labels {out_order[np.logical_not(log_array)]} in `out_order` "
         f"do not appear in network structure."))

  unique_out, cnts_out = np.unique(out_order, return_counts=True)
  if np.any(cnts_out > 1):
    msg = [reverse_mapping[u] for u in unique_out[cnts_out > 1]]
    raise ValueError((f"output labels {msg}"
                      f" appear more than once in `out_order`."))

  if not np.array_equal(np.sort(out_order), np.unique(neg_cons)):
    msg1 = [reverse_mapping[o] for o in out_order]
    msg2 = [reverse_mapping[o] for o in neg_cons]
    raise ValueError((f"{msg1} is not a valid output order "
                      f"for free output indices {msg2}."))

  if len(np.nonzero(unique == 0)[0]) != 0:
    raise ValueError("only nonzero values are allowed "
                     "to specify network structure.")

  if not np.all(pos_cnts == 2):
    msg = [reverse_mapping[o] for o in pos_connections[pos_cnts != 2]]
    raise ValueError(f"contracted connections {msg}"
                     f" do not appear exactly twice.")
  if not np.all(neg_cnts == 1):
    msg = [reverse_mapping[o] for o in neg_connections[neg_cnts != 1]]
    raise ValueError(f"output connections " f"{msg} appear " f"more than once.")

  un_pos = np.unique(pos_cons)
  for u in un_pos:
    dims = flat_dimensions[flat_connections == u]
    if dims[0] != dims[1]:
      raise ValueError(f"tensor dimensions of "
                       f"connection {reverse_mapping[u]}"
                       f" do not match, got dimensions {dims}.")


def _partial_trace(tensor, labels, backend_obj):
  unique, cnts = np.unique(labels, return_counts=True)
  if np.any(cnts == 2):
    shape = backend_obj.shape_tuple(tensor)
    unique_multiple = unique[cnts == 2]
    num_cont = unique_multiple.shape[0]
    ix, iy = np.nonzero(labels[:, None] == unique_multiple)
    trace_label_positions = [ix[iy == n] for n in range(num_cont)]
    contracted_indices = np.array(trace_label_positions).reshape(
        num_cont * 2, order='F')
    free_indices = np.delete(
        np.arange(tensor.ndim, dtype=np.int16), contracted_indices)

    contracted_dimension = np.prod(
        [shape[d] for d in contracted_indices[:num_cont]])
    temp_shape = tuple([contracted_dimension, contracted_dimension])
    result = backend_obj.trace(
        backend_obj.reshape(
            backend_obj.transpose(
                tensor, tuple(np.append(free_indices, contracted_indices))),
            temp_shape))
    new_labels = np.delete(labels, contracted_indices)
    contracted_labels = np.unique(labels[contracted_indices])

    return result, new_labels, contracted_labels
  return tensor, labels, []


def _jittable_ncon(tensors, network_structure, con_order, out_order,
                   backend_obj):
  """Jittable Ncon function.

  Args:
    tensors: List of tensors.
    network_structure: List of list of integers that descripes the network
      structure.
    con_order: Order of the contraction.
    out_order: Order of the final axis order.
    backend_obj: A backend object.

  Returns:
    The final tensor after contraction.
  """

  if not isinstance(tensors, list):
    raise ValueError("`tensors` is not a list")

  if not isinstance(network_structure, list):
    raise ValueError("`network_structure` is not a list")
  network_structure = [np.array(l) for l in network_structure]
  flat_connections = np.concatenate(network_structure)

  if con_order is None:
    con_order = np.unique(flat_connections[flat_connections > 0])
  else:
    con_order = np.array(con_order)

  # partial trace
  for n, tensor in enumerate(tensors):
    tensors[n], network_structure[n], contracted_labels = _partial_trace(
        tensor, network_structure[n], backend_obj)
    con_order = np.delete(
        con_order,
        np.intersect1d(con_order, contracted_labels, return_indices=True)[1])

  # binary contractions
  while len(con_order) > 0:
    cont_ind = con_order[0]  # the next index to be contracted
    locs = np.sort(
        np.nonzero([np.isin(cont_ind, labels) for labels in network_structure
                   ])[0])
    t2 = tensors.pop(locs[1])
    t1 = tensors.pop(locs[0])
    labels_t2 = network_structure.pop(locs[1])
    labels_t1 = network_structure.pop(locs[0])

    common_labels, t1_cont, t2_cont = np.intersect1d(
        labels_t1, labels_t2, assume_unique=True, return_indices=True)

    ind_sort = np.argsort(t1_cont)
    tensors.append(
        backend_obj.tensordot(
            t1, t2, axes=(tuple(t1_cont[ind_sort]), tuple(t2_cont[ind_sort]))))
    network_structure.append(
        np.append(np.delete(labels_t1, t1_cont), np.delete(labels_t2, t2_cont)))

    # remove contracted labels from con_order
    con_order = np.delete(
        con_order,
        np.intersect1d(
            con_order, common_labels, assume_unique=True,
            return_indices=True)[1])

  # outer products
  while len(tensors) > 1:
    t2 = tensors.pop()
    t1 = tensors.pop()
    labels_t2 = network_structure.pop()
    labels_t1 = network_structure.pop()

    tensors.append(backend_obj.outer_product(t1, t2))
    network_structure.append(np.append(labels_t1, labels_t2))

  # final permutation
  if len(network_structure[0]) > 0:
    if out_order is None:
      return backend_obj.transpose(tensors[0], tuple(np.argsort(-out_order)))
    _, l1, l2 = np.intersect1d(
        network_structure, out_order, assume_unique=True, return_indices=True)
    return backend_obj.transpose(tensors[0], tuple(l1[l2]))
  return tensors[0]


def ncon(
    tensors: Any,
    network_structure: Sequence[Sequence],
    con_order: Optional[Sequence] = None,
    out_order: Optional[Sequence] = None,
    check_network: bool = True,
    backend: Optional[Union[Text, BaseBackend]] = None
) -> Union[network_components.BaseNode, Tensor]:
  r"""Contracts a list of tensors or nodes according to a tensor network 
    specification.

    The network is provided as a list of lists, one for each
    tensor, specifying labels for the edges connected to that tensor.

    If a contraction order `con_order` and an output order `out_order`
    are both provided, the edge labels can be anything.
    Otherwise (`con_order == None or out_order == None`), the edge labels 
    must be nonzero integers and edges will be contracted in ascending order.
    Negative integers denote the (dangling) indices of the output tensor,
    which will be in descending order, e.g. `[-1,-2,-3,...]`.

    For example, matrix multiplication:

    .. code-block:: python

      A = np.array([[1.0, 2.0], [3.0, 4.0]])
      B = np.array([[1.0, 1.0], [0.0, 1.0]])
      ncon([A,B], [(-1, 1), (1, -2)])

    Matrix trace:

    .. code-block:: python

      A = np.array([[1.0, 2.0], [3.0, 4.0]])
      ncon([A], [(1, 1)]) # 5.0

    Note: 
      The reason `0` is not allowed as an edge label without manually
      specifying the contraction order is to maintain compatibility with the
      `original NCON implementation`_. However, the use of `0` in `con_order` 
      to denote outer products is not (currently) 
      supported in this implementation.
    
    .. _original NCON implementation:
      https://arxiv.org/abs/1402.0939

    Args:
      tensors: List of `Tensors` or `BaseNodes`.
      network_structure: List of lists specifying the tensor network structure.
      con_order: List of edge labels specifying the contraction order.
      out_order: List of edge labels specifying the output order.
      check_network: Boolean flag. If `True` check the network.
      backend: String specifying the backend to use. Defaults to
        `tensornetwork.backend_contextmanager.get_default_backend`.

    Returns:
      The result of the contraction. The result is returned as a `Node`
      if all elements of `tensors` are `BaseNode` objects, else
      it is returned as a `Tensor` object.
    """
  if backend is None:
    backend = get_default_backend()
  if isinstance(backend, BaseBackend):
    backend_obj = backend
  else:
    backend_obj = backend_factory.get_backend(backend)

  if out_order == []:  #allow empty list as input
    out_order = None
  if con_order == []:  #allow empty list as input
    con_order = None

  are_nodes = [isinstance(t, network_components.BaseNode) for t in tensors]
  nodes = {t for t in tensors if isinstance(t, network_components.BaseNode)}
  if not all([n.backend.name == backend_obj.name for n in nodes]):
    raise ValueError("Some nodes have backends different from '{}'".format(
        backend_obj.name))

  _tensors = []
  for t in tensors:
    if isinstance(t, network_components.BaseNode):
      _tensors.append(t.tensor)
    else:
      _tensors.append(t)
  _tensors = [backend_obj.convert_to_tensor(t) for t in _tensors]

  network_structure = [np.array(l) for l in network_structure]
  flat_connections = np.concatenate(network_structure)
  mapping = None
  reverse_mapping = None

  if flat_connections.dtype.type is np.str_:
    if (con_order is not None) and (np.array(con_order).dtype.type
                                    is not np.str_):
      raise TypeError("network_structure and con_order "
                      "have to have the same dtype")
    if (out_order is not None) and (np.array(out_order).dtype.type
                                    is not np.str_):
      raise TypeError("network_structure and out_order "
                      "have to have the same dtype")
    network_structure, mapping = _map_string_to_int(network_structure)
    
    reverse_mapping = {v: k for k, v in mapping.items()}
    flat_connections = np.concatenate(network_structure)
  if out_order is None:
    out_order = np.sort(flat_connections[flat_connections < 0])[::-1]
  else:
    if mapping is not None:
      l = []
      for o in out_order:
        try:
          l.append(mapping[o])
        except KeyError:
          raise KeyError(
              f"output label '{o}' does not appear in network_structure")
      out_order = np.array(l)
    else:
      out_order = np.array(out_order)
  if con_order is None:
    con_order = np.unique(flat_connections[flat_connections > 0])
  else:
    if mapping is not None:
      l = []
      for o in con_order:
        try:
          l.append(mapping[o])
        except KeyError:
          raise KeyError(
              f"contraction label '{o}' does not appear in network_structure")
      con_order = np.array(l)
    else:
      con_order = np.array(con_order)
  if check_network:
    _check_network(network_structure, [t.shape for t in _tensors],
                   flat_connections, con_order, out_order, reverse_mapping)
  if backend not in _CACHED_JITTED_NCONS:
    _CACHED_JITTED_NCONS[backend] = backend_obj.jit(
        _jittable_ncon, static_argnums=(1, 2, 3, 4))
  res_tensor = _CACHED_JITTED_NCONS[backend](_tensors, network_structure,
                                             con_order, out_order, backend_obj)
  if all(are_nodes):
    return network_components.Node(res_tensor, backend=backend_obj)
  return res_tensor


def ncon_network(
    tensors: Sequence[Tensor],
    network_structure: Sequence[Sequence],
    con_order: Optional[Sequence] = None,
    out_order: Optional[Sequence] = None,
    backend: Optional[Union[Text, BaseBackend]] = None
) -> Tuple[List[network_components.BaseNode], List[network_components.Edge],
           List[network_components.Edge]]:
  r"""
    Deprecated: support of this function will bbe dropped without futher notice.

    Creates a network from a list of tensors according to `tensors`.

    The network is provided as a list of lists, one for each
    tensor, specifying labels for the edges connected to that tensor.

    If a contraction order `con_order` and an output order `out_order`
    are both provided, the edge labels can be anything.
    Otherwise (`con_order == None or out_order == None`), the edge labels 
    must be integers and edges will be contracted in ascending order.
    Negative integers denote the (dangling) indices of the output tensor,
    which will be in descending order, e.g. `[-1,-2,-3,...]`.

    This is used internally by `ncon()`.

    Args:
      tensors: List of `Tensor`s.
      network_structure: List of lists specifying the tensor network.
      con_order: List of edge labels specifying the contraction order.
      out_order: List of edge labels specifying the output order.
      backend: String or BaseBackend object specifying the backend to use. 
        Defaults to the default TensorNetwork backend.
    Returns:
      nodes: List of constructed nodes in the same order as given in `tensors`.
      con_edges: List of internal `Edge` objects in contraction order.
      out_edges: List of dangling `Edge` objects in output order.
  """
  if len(tensors) != len(network_structure):
    raise ValueError('len(tensors) != len(network_structure)')

  nodes, edges = _build_network(tensors, network_structure, backend)

  if con_order is None:
    try:
      con_order = sorted((k for k in edges if k >= 0))
      if con_order and con_order[0] == 0:
        raise ValueError("'0' is not a valid edge label when the "
                         "contraction order is not specified separately.")
    except TypeError:
      raise ValueError("Non-integer edge label(s): {}".format(
          list(edges.keys())))
  else:
    if len(con_order) != len(set(con_order)):
      raise ValueError("Duplicate labels in con_order: {}".format(con_order))

  if out_order is None:
    try:
      out_order = sorted((k for k in edges if k < 0), reverse=True)
    except TypeError:
      raise ValueError("Non-integer edge label(s): {}".format(
          list(edges.keys())))
  else:
    if len(out_order) != len(set(out_order)):
      raise ValueError("Duplicate labels in out_order: {}".format(out_order))

  try:
    con_edges = [edges[k] for k in con_order]
    out_edges = [edges[k] for k in out_order]
  except KeyError as err:
    raise ValueError("Order contained an unknown edge label: {}".format(
        err.args[0]))

  if len(con_edges) + len(out_edges) != len(edges):
    raise ValueError(
        "Edges {} were not included in the contraction and output "
        "ordering.".format(
            list(set(edges.keys()) - set(con_order) - set(out_order))))

  for e in con_edges:
    if e.is_dangling():
      raise ValueError(
          "Contraction edge {} appears only once in the network.".format(
              str(e)))

  for e in out_edges:
    if not e.is_dangling():
      raise ValueError(
          "Output edge {} appears more than once in the network.".format(
              str(e)))

  return nodes, con_edges, out_edges


def _build_network(
    tensors: Sequence[Tensor],
    network_structure: Sequence[Sequence],
    backend: Optional[Union[BaseBackend, Text]] = None,
) -> Tuple[List[network_components.BaseNode], Dict[Any,
                                                   network_components.Edge]]:
  nodes = []
  edges = {}
  for i, (tensor, edge_lbls) in enumerate(zip(tensors, network_structure)):
    if len(tensor.shape) != len(edge_lbls):
      raise ValueError(
          "Incorrect number of edge labels specified tensor {}".format(i))
    if isinstance(tensor, network_components.BaseNode):
      node = tensor
    else:
      node = network_components.Node(
          tensor, name="tensor_{}".format(i), backend=backend)

    nodes.append(node)

    for (axis_num, edge_lbl) in enumerate(edge_lbls):
      if edge_lbl not in edges:
        e = node[axis_num]
        e.set_name(str(edge_lbl))
        edges[edge_lbl] = e
      else:
        # This will raise an error if the edges are not dangling.
        e = network_components.connect(
            edges[edge_lbl], node[axis_num], name=str(edge_lbl))
        edges[edge_lbl] = e
  return nodes, edges
