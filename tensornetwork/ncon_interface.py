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
from typing import Any, Sequence, List, Optional, Union, Text, Tuple
from tensornetwork import network_components
from tensornetwork.backend_contextmanager import get_default_backend
from tensornetwork.backends import backend_factory
from tensornetwork.backends.abstract_backend import AbstractBackend
Tensor = Any

_CACHED_JITTED_NCONS = {}


def _get_cont_out_labels(
    network_structure: List[List]) -> Tuple[List, List, List, List]:
  """
  Compute the contracted and free labels of `network_structure`.
  Contracted labels are labels appearing more than once,
  free labels are labels appearing exactly once.
  Computed lists are ordered according to int and ASCII ordering 
  for integer and string values, with first entries in each list
  being ordered integer labels followed by ASCII ordered string 
  labels.
  Returns:
    cont_labels, out_labels: The contracted and free labels
      of `network_structure`.
  """
  flat_labels = [l for sublist in network_structure for l in sublist]
  out_labels = [l for l in flat_labels if flat_labels.count(l) == 1]
  int_out_labels = sorted([o for o in out_labels if not isinstance(o, str)
                          ])[::-1]
  # pylint: disable=unnecessary-lambda
  str_out_labels = sorted([o for o in out_labels if isinstance(o, str)],
                          key=lambda x: str(x))

  cont_labels = []
  for l in flat_labels:
    if (flat_labels.count(l) > 1) and (l not in cont_labels):
      cont_labels.append(l)
  int_cont_labels = sorted([o for o in cont_labels if not isinstance(o, str)])
  # pylint: disable=unnecessary-lambda
  str_cont_labels = sorted([o for o in cont_labels if isinstance(o, str)],
                           key=lambda x: str(x))

  return int_cont_labels, str_cont_labels, int_out_labels, str_out_labels


def _canonicalize_network_structure(cont_labels, out_labels, network_structure):
  mapping = dict(zip(cont_labels, np.arange(1, len(cont_labels) + 1)))
  out_mapping = dict(zip(out_labels, -np.arange(1, len(out_labels) + 1)))
  mapping.update(out_mapping)
  mapped_network_structure = [
      [mapping[label] for label in labels] for labels in network_structure
  ]
  return mapped_network_structure, mapping


def _check_network(network_structure, tensor_dimensions, con_order, out_order):
  if len(network_structure) != len(tensor_dimensions):
    raise ValueError("number of tensors does not match the"
                     " number of network connections.")

  for n, dims in enumerate(tensor_dimensions):
    if len(dims) != len(network_structure[n]):
      raise ValueError(f"number of indices does not match"
                       f" number of labels on tensor {n}.")
  flat_labels = [l for sublist in network_structure for l in sublist]
  tmp_wrong_labels = [l for l in flat_labels if flat_labels.count(l) > 2]
  wrong_labels = []
  for l in tmp_wrong_labels:
    if l not in wrong_labels:
      wrong_labels.append(l)

  if len(wrong_labels) != 0:
    raise ValueError(
        f"labels {wrong_labels} appear more than twice in `network_structure`.")
  # pylint: disable=line-too-long
  int_cont_labels, str_cont_labels, int_out_labels, str_out_labels = _get_cont_out_labels(
      network_structure)
  out_labels = int_out_labels + str_out_labels
  cont_labels = int_cont_labels + str_cont_labels

  if (int_cont_labels.count(0) > 0) or (int_out_labels.count(0) > 0):
    raise ValueError("only nonzero values are allowed to "
                     "specify network structure.")

  if (out_order is None) and (not np.all([o[0] == '-'
                                          for o in str_out_labels])):
    raise ValueError(f"open string labels have to be prepended with '-'; "
                     f"found {out_labels}")
  if (out_order is None) and (not np.all(np.array(int_out_labels) < 0)):
    raise ValueError(f"open integer labels have to be negative integers, "
                     f"found {out_labels}")
  if np.any(np.array(int_cont_labels) < 0):
    raise ValueError(
        f"contracted labels can only be positive integers or strings"
        f", found {cont_labels}.")
  if np.any([l[0] == '-' for l in str_cont_labels]):
    raise ValueError(f"contracted labels must not be prepended with '-'"
                     f", found {cont_labels}.")

  if con_order is not None:
    if len(con_order) != len(cont_labels):
      raise ValueError(f"`con_order = {con_order} is not "
                       f"a valid contraction order for contracted "
                       f"labels {cont_labels}")
    # check if all labels on con_order appear in network_structure
    labels = [o for o in con_order if o not in flat_labels]
    if len(labels) != 0:
      raise ValueError(f"labels {labels} in `con_order` do not appear as "
                       f"contracted labels in `network_structure`.")
    labels = [o for o in con_order if o not in cont_labels]
    if len(labels) != 0:
      raise ValueError(f"labels {labels} in `con_order` appear only "
                       f"once in `network_structure`.")
    for l in con_order:
      if con_order.count(l) != 1:
        raise ValueError(f"label {l} appears more than once in `con_order`.")

  if out_order is not None:
    if len(out_order) != len(out_labels):
      raise ValueError(f"`out_order` = {out_order} is not "
                       f"a valid output order for open "
                       f"labels {out_labels}")

    # check if all labels on out_order appear in network_structure
    labels = [o for o in out_order if o not in flat_labels]
    if len(labels) != 0:
      raise ValueError(f"labels {labels} in `out_order` do not "
                       f"appear in `network_structure`.")
    labels = [o for o in out_order if o not in out_labels]
    if len(labels) != 0:
      raise ValueError(f"labels {labels} in `out_order` appear more than "
                       f"once in `network_structure`.")
    for l in out_order:
      if out_order.count(l) != 1:
        raise ValueError(f"label {l} appears more than once in `out_order`.")

  return cont_labels, out_labels


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
    temp_shape = tuple([shape[pos] for pos in free_indices] +
                       [contracted_dimension, contracted_dimension])
    result = backend_obj.trace(
        backend_obj.reshape(
            backend_obj.transpose(
                tensor, tuple(np.append(free_indices, contracted_indices))),
            temp_shape))
    new_labels = np.delete(labels, contracted_indices)
    contracted_labels = np.unique(labels[contracted_indices])

    return result, new_labels, contracted_labels
  return tensor, labels, []


def _jittable_ncon(tensors, flat_labels, sizes, con_order, out_order,
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
  # some jax-juggling to avoid retracing ...
  slices = np.append(0, np.cumsum(sizes))
  network_structure = [
      np.array(flat_labels)[slices[n]:slices[n + 1]]
      for n in range(len(slices) - 1)
  ]
  con_order = np.array(con_order)
  out_order = np.array(out_order)

  # now we're ready to do stuff
  if not isinstance(tensors, list):
    raise ValueError("`tensors` is not a list")

  if not isinstance(network_structure, list):
    raise ValueError("`network_structure` is not a list")

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
    i1, i2 = np.nonzero(out_order[:, None] == network_structure[0][None, :])
    return backend_obj.transpose(tensors[0], tuple(i1[i2]))
  return tensors[0]


def ncon(
    tensors: Sequence[Union[network_components.AbstractNode, Tensor]],
    network_structure: Sequence[Sequence],
    con_order: Optional[Sequence] = None,
    out_order: Optional[Sequence] = None,
    check_network: bool = True,
    backend: Optional[Union[Text, AbstractBackend]] = None
) -> Union[network_components.AbstractNode, Tensor]:
  r"""Contracts a list of tensors or nodes according to a tensor network 
    specification.

    The network is provided as a list of lists, one for each
    tensor, specifying labels for the edges connected to that tensor.
    
    Labels appearing only once in `network_structure` (open labels)
    remain uncontracted, labels appearing twice (contracted labels) are
    contracted over. 
    If `out_order = None`, output labels can either be negative numbers or
    strings with a hyphen character ('-') prepended, e.g. '-out_label_1'.
    If `out_order = None` output labels are ordered according to descending
    number ordering and ascending ASCII ordering, with number labels always 
    appearing before string labels. Example:
    network_structure = [[-1, 1, '-3', '2'], [-2, '2', 1, '-33']] results 
    in an output order of [-1, -2, '-3', '-33'].
    If `out_order` is given, the indices of the resulting tensor will be
    transposed into this order. In this case output labels can be arbitrary
    numbers and arbitrary strings (no minus or hyphen necessary).
    
    If `con_order = None`, `ncon` will first contract all number labels 
    in ascending order followed by all string labels in ascending ASCII 
    order.
    If `con_order` is given, `ncon` will contract according to this order.

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
      tensors: List of `Tensors` or `AbstractNodes`.
      network_structure: List of lists specifying the tensor network structure.
      con_order: List of edge labels specifying the contraction order.
      out_order: List of edge labels specifying the output order.
      check_network: Boolean flag. If `True` check the network.
      backend: String specifying the backend to use. Defaults to
        `tensornetwork.backend_contextmanager.get_default_backend`.

    Returns:
      The result of the contraction. The result is returned as a `Node`
      if all elements of `tensors` are `AbstractNode` objects, else
      it is returned as a `Tensor` object.
    """
  if backend is None:
    backend = get_default_backend()
  if isinstance(backend, AbstractBackend):
    backend_obj = backend
  else:
    backend_obj = backend_factory.get_backend(backend)

  if out_order == []:  #allow empty list as input
    out_order = None
  if con_order == []:  #allow empty list as input
    con_order = None

  # convert to lists
  network_structure = [list(l) for l in network_structure]
  are_nodes = [isinstance(t, network_components.AbstractNode) for t in tensors]
  nodes = {t for t in tensors if isinstance(t, network_components.AbstractNode)}
  if not all([n.backend.name == backend_obj.name for n in nodes]):
    raise ValueError("Some nodes have backends different from '{}'".format(
        backend_obj.name))

  _tensors = []
  for t in tensors:
    if isinstance(t, network_components.AbstractNode):
      _tensors.append(t.tensor)
    else:
      _tensors.append(t)
  _tensors = [backend_obj.convert_to_tensor(t) for t in _tensors]
  if check_network:
    cont_labels, out_labels = _check_network(network_structure,
                                             [t.shape for t in _tensors],
                                             con_order, out_order)
  else:
    # map the network strcuture to integers; if any of the labels is a `str`
    # type, the ordering defaults to string-ordering, i.e.
    # [[1, 2, '12'], [1, 9]] -> [[1, -2, -1],[1, -3]]
    # pylint: disable=line-too-long
    int_cont_labels, str_cont_labels, int_out_labels, str_out_labels = _get_cont_out_labels(
        network_structure)
    cont_labels = int_cont_labels + str_cont_labels
    out_labels = int_out_labels + str_out_labels
  network_structure, mapping = _canonicalize_network_structure(
      cont_labels, out_labels, network_structure)

  network_structure = [np.array(l) for l in network_structure]
  flat_labels = np.concatenate(network_structure)
  if out_order is None:
    out_order = np.sort(flat_labels[flat_labels < 0])[::-1]
  else:
    l = []
    for o in out_order:
      l.append(mapping[o])
    out_order = np.array(l)
  if con_order is None:
    #canonicalization of network structure takes care of appropriate
    #contraction ordering (i.e. use ASCII ordering for str and
    #regular ordering for int)
    con_order = np.unique(flat_labels[flat_labels > 0])
  else:
    l = []
    for o in con_order:
      l.append(mapping[o])
    con_order = np.array(l)

  if backend not in _CACHED_JITTED_NCONS:
    _CACHED_JITTED_NCONS[backend] = backend_obj.jit(
        _jittable_ncon, static_argnums=(1, 2, 3, 4, 5))
  # we need to pack everything into tuples, or jax will insist on retracing ...
  sizes = tuple([len(l) for l in network_structure])
  res_tensor = _CACHED_JITTED_NCONS[backend](_tensors, tuple(flat_labels),
                                             sizes, tuple(con_order),
                                             tuple(out_order), backend_obj)
  if all(are_nodes):
    return network_components.Node(res_tensor, backend=backend_obj)
  return res_tensor
