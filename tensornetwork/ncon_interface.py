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
from typing import Any, Sequence, List, Optional, Union, Text, Tuple, Dict, Set
from tensornetwork import tensor as tn_tensor
from tensornetwork.backend_contextmanager import get_default_backend
from tensornetwork.backends import backend_factory
from tensornetwork.backends.abstract_backend import AbstractBackend
import time
Tensor = Any

_CACHED_JITTED_NCONS = {}


def _get_cont_out_labels(
    network_structure: Sequence[Sequence[Union[int, str]]]) -> Any:
  """
  Compute the contracted and free and labels of `network_structure`, 
  using the following rules:
  * Any negative number-type element and any hyphen-prepended str-type 
    element are considered output labels.
  * Any positive number-type element and any non-hyphen-prepended 
    str-type element are considered contracted labels.
  * Any negative number-type element appearing more than once, any 
    hyphen-prepended str-type element appearing more than once, 
    any positive element appearing exactly once and 
    any element appearing more than twice are considered batch labels.

  Computed lists are ordered according to int and ASCII ordering 
  for integer and string values, with first entries in each list
  being ordered integer labels followed by ASCII ordered string 
  labels.

  Returns:
    int_cont_labels: The number-type contracted labels
    str_cont_labels: The str-type contracted labels
    int_out_labels: The number-type output labels
    str_out_labels: The str-type output labels
  """
  flat_labels = [l for sublist in network_structure for l in sublist]
  int_labels = {o for o in flat_labels if not isinstance(o, str)}
  str_labels = {o for o in flat_labels if isinstance(o, str)}

  int_out_labels = sorted([l for l in int_labels if l < 0], reverse=True)
  int_cont_labels = sorted([label for label in int_labels if label >= 0])
  # pylint: disable=unnecessary-lambda
  str_out_labels = sorted([label for label in str_labels if label[0] == '-'],
                          key=lambda x: str(x))
  # pylint: disable=unnecessary-lambda
  str_cont_labels = sorted([label for label in str_labels if label[0] != '-'],
                           key=lambda x: str(x))
  # pylint: disable=line-too-long
  return int_cont_labels, str_cont_labels, int_out_labels, str_out_labels


def _canonicalize_network_structure(
    network_structure: Sequence[Sequence[Union[int, str]]]
) -> Tuple[List[List], Dict]:
  """
  Map `network_structure` to a canonical form. 
  The elements in `network_structure` are replaced
  by integers according to the following rules:
  1. All negative numbers are sorted in decreasing order and mapped to 
     to decreasing integers, starting with -1.
     E.g., the numbers [-4,-10,-1] are mapped to 
     -1 -> -1, -4 -> -2, -10 -> -3
  2. All strings prepended with a hyphen '-' are ordered increasingly
     using ASCII ordering, and mapped to decreasing negative integers
     starting with the next value following the last integer under  
     point 1. above. E.g. [-4,-10,-1,'-303','-a','-33']
     is mapped to 
     -1 -> -1, -4 -> -2, -10 -> -3, '-303' -> -4, '-33' -> -5, '-a' -> -6
  3. All positive numbers are sorted increasingly and mapped to 
     increasing integers, starting at 1
  4. All strings without a prepended hyphen are sorted increasingly
     using ASCII order and mapped to positive integers, starting
     with the next integer value following the last used value under
     point 3. 
  
  """
  flat_labels = [l for sublist in network_structure for l in sublist]
  neg_int_labels = sorted(
      list({l for l in flat_labels if not isinstance(l, str) and l < 0}))
  pos_int_labels = sorted(
      list({l for l in flat_labels if not isinstance(l, str) and l > 0}))
  neg_str_labels = sorted(
      {l for l in flat_labels if isinstance(l, str) and l[0] == '-'},
      reverse=True)
  pos_str_labels = sorted(
      list({l for l in flat_labels if isinstance(l, str) and l[0] != '-'}))

  neg_mapping = dict(
      zip(neg_str_labels + neg_int_labels,
          np.arange(-len(neg_int_labels + neg_str_labels), 0)))
  pos_mapping = dict(
      zip(pos_int_labels + pos_str_labels,
          np.arange(1, 1 + len(pos_int_labels + pos_str_labels))))
  neg_mapping.update(pos_mapping)
  mapped_network_structure = [
      [neg_mapping[label] for label in labels] for labels in network_structure
  ]
  return mapped_network_structure, neg_mapping


def _check_network(network_structure: Sequence[Sequence[Union[int, str]]],
                   tensor_dimensions: List[Tuple[int]],
                   con_order: Optional[List[Union[int, str]]] = None,
                   out_order: Optional[List[Union[int, str]]] = None) -> None:
  """
  Perform checks on `network_structure`.
  """
  # check if number of tensors matches the number of lists
  # in network_structure
  if len(network_structure) != len(tensor_dimensions):
    raise ValueError("number of tensors does not match the"
                     " number of network connections.")

  # check number of labels of each element in network_structure
  # matches the tensor order
  for n, dims in enumerate(tensor_dimensions):
    if len(dims) != len(network_structure[n]):
      raise ValueError(f"number of indices does not match"
                       f" number of labels on tensor {n}.")

  flat_labels = [l for sublist in network_structure for l in sublist]
  # pylint: disable=line-too-long
  int_cont_labels, str_cont_labels, int_out_labels, str_out_labels = _get_cont_out_labels(
      network_structure)
  out_labels = int_out_labels + str_out_labels
  cont_labels = int_cont_labels + str_cont_labels
  str_labels = str_cont_labels + [l[1:] for l in str_out_labels]
  mask = [l.isalnum() for l in str_labels]
  if not np.all(mask):
    raise ValueError(
        f"only alphanumeric values allowed for string labels, "
        f"found {[l for n, l in enumerate(str_labels) if not mask[n]]}")
  # make sure no value 0 is used as a label (legacy behaviour)
  if int_cont_labels.count(0) > 0:
    raise ValueError("only nonzero values are allowed to "
                     "specify network structure.")

  if con_order is not None:
    #check that all integer elements in `con_order` are positive
    int_cons = [o for o in con_order if not isinstance(o, str)]
    labels = [o for o in int_cons if o < 0]
    if len(labels) > 0:
      raise ValueError(f"all number type labels in `con_order` have "
                       f"to be positive, found {labels}")
    str_cons = [o for o in con_order if isinstance(o, str)]
    labels = [o for o in str_cons if o[0] == '-']
    #check that all string type elements in `con_order` have no hyphens
    if len(labels) > 0:
      raise ValueError(f"all string type labels in `con_order` "
                       f"must be unhyphenized, found {labels}")

    # check that elements in `con_order` appear only once
    labels = []
    for l in con_order:
      if (con_order.count(l) != 1) and (l not in labels):
        labels.append(l)
    if len(labels) > 0:
      raise ValueError(f"labels {labels} appear more than once in `con_order`.")

    # check if passed `con_order` makes sense
    if len(con_order) != len(cont_labels):
      raise ValueError(f"`con_order = {con_order} is not "
                       f"a valid contraction order for contracted "
                       f"labels {cont_labels}")

    # check if all labels in `con_order` appear in `network_structure`
    labels = [o for o in con_order if o not in flat_labels]
    if len(labels) != 0:
      raise ValueError(f"labels {labels} in `con_order` do not appear as "
                       f"contracted labels in `network_structure`.")

  if out_order is not None:
    #check that all integer elements in `out_order` are negative
    int_outs = [o for o in out_order if not isinstance(o, str)]
    labels = [o for o in int_outs if o > 0]
    if len(labels) > 0:
      raise ValueError(f"all number type labels in `out_order` have "
                       f"to be negative, found {labels}")

    #check that all string type elements in `out_order` have hyphens
    str_outs = [o for o in out_order if isinstance(o, str)]
    labels = [o for o in str_outs if o[0] != '-']
    if len(labels) > 0:
      raise ValueError(f"all string type labels in `out_order` "
                       f"have to be hyphenized, found {labels}")

    # check that all elements in `out_order` appear exactly once
    labels = []
    for l in out_order:
      if (out_order.count(l) != 1) and (l not in labels):
        labels.append(l)
    if len(labels) > 0:
      raise ValueError(f"labels {labels} appear more than once in `out_order`.")

    # check if `out_order` has right length
    if len(out_order) != len(out_labels):
      raise ValueError(f"`out_order` = {out_order} is not "
                       f"a valid output order for open "
                       f"labels {out_labels}")

    # check if all labels in `out_order` appear in `network_structure`
    labels = [o for o in out_order if o not in flat_labels]
    if len(labels) != 0:
      raise ValueError(f"labels {labels} in `out_order` do not "
                       f"appear in `network_structure`.")

  # check if contracted dimensions are matching
  mismatched_labels = []
  for l in cont_labels:
    dims = {
        tensor_dimensions[m][n]
        for m, labels in enumerate(network_structure)
        for n, l1 in enumerate(labels)
        if l1 == l
    }
    if len(dims) > 1:
      mismatched_labels.append(l)

  if len(mismatched_labels) > 0:
    raise ValueError(
        f"tensor dimensions for labels {mismatched_labels} are mismatching")


def _partial_trace(
    tensor: Tensor, labels: List,
    backend_obj: AbstractBackend) -> Tuple[Tensor, List, List]:
  """
  Perform the partial trace of `tensor`.
  All labels appearing twice in `labels` are traced out.
  Argns:
    tensor: A tensor.
    labels: The ncon-style labels of `tensor`.
  Returns:
    Tensor: The result of the tracing.
  """
  trace_labels = [l for l in labels if labels.count(l) == 2]
  if len(trace_labels) > 0:
    num_cont = len(trace_labels) // 2
    unique_trace_labels = sorted(trace_labels)[0:-1:2]
    trace_label_positions = [[
        n for n, label in enumerate(labels) if label == trace_label
    ] for trace_label in unique_trace_labels]
    contracted_indices = [l[0] for l in trace_label_positions
                         ] + [l[1] for l in trace_label_positions]
    free_indices = [
        n for n in range(len(labels)) if n not in contracted_indices
    ]
    shape = backend_obj.shape_tuple(tensor)
    contracted_dimension = np.prod(
        [shape[d] for d in contracted_indices[:num_cont]])
    temp_shape = tuple([shape[pos] for pos in free_indices] +
                       [contracted_dimension, contracted_dimension])
    result = backend_obj.trace(
        backend_obj.reshape(
            backend_obj.transpose(tensor,
                                  tuple(free_indices + contracted_indices)),
            temp_shape))
    new_labels = [l for l in labels if l not in unique_trace_labels]
    return result, new_labels, unique_trace_labels
  return tensor, labels, []


def _batch_cont(
    t1: Tensor, t2: Tensor, tensors: List[Tensor],
    network_structure: List[List], con_order: List, common_batch_labels: Set,
    labels_t1: List, labels_t2: List, backend_obj: AbstractBackend
) -> Tuple[Tensor, List[List], List]:
  """
  Subroutine for performing a batched contraction of tensors `t1` and `t2`.
  Args:
    t1: A Tensor.
    t2: A Tensor.
    tensors: List of Tensor objects.
    network_structure: The canonical labels of the networks.
    con_order: Array of contracted labels.
    common_batch_labels: The common batch labels of `t1` and `t2`.
    labels_t1: The labels of `t1`
    labels_t2: The labels of `t2`
    backend_obj: A backend object.
  Returns:
    List[Tensor]: Updated list of tensors.
    List[List]: Updated `network_structure`.
    List: Updated `con_order` (contraction order).
  """
  common_batch_labels = list(common_batch_labels)
  #find positions of common batch labels
  t1_batch_pos = [labels_t1.index(l) for l in common_batch_labels]
  t2_batch_pos = [labels_t2.index(l) for l in common_batch_labels]
  #find positions of contracted non-batch labels
  non_batch_labels_t1 = {l for l in labels_t1 if l not in common_batch_labels}
  non_batch_labels_t2 = {l for l in labels_t2 if l not in common_batch_labels}

  common_contracted_labels = list(
      non_batch_labels_t1.intersection(non_batch_labels_t2))
  t1_cont = [labels_t1.index(l) for l in common_contracted_labels]
  t2_cont = [labels_t2.index(l) for l in common_contracted_labels]

  free_labels_t1 = set(labels_t1) - set(common_contracted_labels) - set(
      common_batch_labels)
  free_labels_t2 = set(labels_t2) - set(common_contracted_labels) - set(
      common_batch_labels)

  # find positions of uncontracted non-batch labels
  free_pos_t1 = [n for n, l in enumerate(labels_t1) if l in free_labels_t1]
  free_pos_t2 = [n for n, l in enumerate(labels_t2) if l in free_labels_t2]

  t1_shape = np.array(backend_obj.shape_tuple(t1))
  t2_shape = np.array(backend_obj.shape_tuple(t2))

  newshape_t1 = (np.prod(t1_shape[t1_batch_pos]),
                 np.prod(t1_shape[free_pos_t1]), np.prod(t1_shape[t1_cont]))
  newshape_t2 = (np.prod(t2_shape[t2_batch_pos]), np.prod(t2_shape[t2_cont]),
                 np.prod(t2_shape[free_pos_t2]))

  #bring batch labels to the front
  order_t1 = tuple(t1_batch_pos + free_pos_t1 + t1_cont)
  order_t2 = tuple(t2_batch_pos + t2_cont + free_pos_t2)

  mat1 = backend_obj.reshape(backend_obj.transpose(t1, order_t1), newshape_t1)
  mat2 = backend_obj.reshape(backend_obj.transpose(t2, order_t2), newshape_t2)
  result = backend_obj.matmul(mat1, mat2)
  final_shape = tuple(
      np.concatenate([
          t1_shape[t1_batch_pos], t1_shape[free_pos_t1], t2_shape[free_pos_t2]
      ]))
  result = backend_obj.reshape(result, final_shape)

  # update labels, tensors, network_structure and con_order
  new_labels = [labels_t1[i] for i in t1_batch_pos] + [
      labels_t1[i] for i in free_pos_t1
  ] + [labels_t2[i] for i in free_pos_t2]

  network_structure.append(new_labels)
  tensors.append(result)
  con_order = [c for c in con_order if c not in common_contracted_labels]

  return tensors, network_structure, con_order


def label_intersection(labels1, labels2):
  common_labels = list(set(labels1).intersection(labels2))
  idx_1 = [labels1.index(l) for l in common_labels]
  idx_2 = [labels2.index(l) for l in common_labels]
  return common_labels, idx_1, idx_2


def _jittable_ncon(tensors: List[Tensor], flat_labels: Tuple[int],
                   sizes: Tuple[int], con_order: Tuple[int],
                   out_order: Tuple[int],
                   backend_obj: AbstractBackend) -> Tensor:
  """
  Jittable Ncon function. Performs the contraction of `tensors`.
  Args:
    tensors: List of tensors.
    flat_labels: A Tuple of integers.
    sizes: Tuple of int used to reconstruct `network_structure` from
      `flat_labels`.
    con_order: Order of the contraction.
    out_order: Order of the final axis order.
    backend_obj: A backend object.

  Returns:
    The final tensor after contraction.
  """
  # some jax-juggling to avoid retracing ...
  flat_labels = list(flat_labels)
  slices = np.append(0, np.cumsum(sizes))
  network_structure = [
      flat_labels[slices[n]:slices[n + 1]] for n in range(len(slices) - 1)
  ]
  out_order = list(out_order)
  con_order = list(con_order)
  # pylint: disable=unnecessary-comprehension
  init_con_order = [c for c in con_order]
  init_network_structure = [c for c in network_structure]

  # partial trace
  for n, tensor in enumerate(tensors):
    tensors[n], network_structure[n], contracted_labels = _partial_trace(
        tensor, network_structure[n], backend_obj)
    if len(contracted_labels) > 0:
      con_order = [c for c in con_order if c not in contracted_labels]

  flat_labels = [l for sublist in network_structure for l in sublist]
  # contracted all positive labels appearing only once in `network_structure`
  contractable_labels = [
      l for l in flat_labels if (flat_labels.count(l) == 1) and (l > 0)
  ]
  # update con_order
  if len(contractable_labels) > 0:
    con_order = [o for o in con_order if o not in contractable_labels]
  # collapse axes of single-labelled tensors
  locs = []
  for n, labels in enumerate(network_structure):
    if len(set(labels).intersection(contractable_labels)) > 0:
      locs.append(n)

  for loc in locs:
    labels = network_structure[loc]
    contractable_inds = [labels.index(l) for l in contractable_labels]
    network_structure[loc] = [l for l in labels if l not in contractable_labels]
    tensors[loc] = backend_obj.sum(tensors[loc], tuple(contractable_inds))

  # perform binary and batch contractions
  skip_counter = 0
  batch_labels = []
  batch_cnts = []
  for l in set(flat_labels):
    cnt = flat_labels.count(l)
    if (cnt > 2) or (cnt == 2 and l < 0):
      batch_labels.append(l)
      batch_cnts.append(cnt)

  while len(con_order) > 0:
    # the next index to be contracted
    cont_ind = con_order[0]
    if cont_ind in batch_labels:
      # if its still a batch index then do it later
      con_order.append(con_order.pop(0))
      skip_counter += 1
      # avoid being stuck in an infinite loop
      if skip_counter > len(con_order):
        raise ValueError(f"ncon seems stuck in an infinite loop. \n"
                         f"Please check if `con_order` = {init_con_order} is "
                         f"a valid contraction order for \n"
                         f"`network_structure` = {init_network_structure}")
      continue

    # find locations of `cont_ind` in `network_structure`
    locs = [
        n for n, labels in enumerate(network_structure) if cont_ind in labels
    ]

    t2 = tensors.pop(locs[1])
    t1 = tensors.pop(locs[0])
    labels_t2 = network_structure.pop(locs[1])
    labels_t1 = network_structure.pop(locs[0])
    common_labels, t1_cont, t2_cont = label_intersection(labels_t1, labels_t2)
    # check if there are batch labels (i.e. labels appearing more than twice
    # in `network_structure`).
    common_batch_labels = set(batch_labels).intersection(common_labels)
    if len(common_batch_labels) > 0:
      # case1: both tensors have one or more common batch indices -> use matmul
      ix = np.nonzero(
          np.array(batch_labels)[:, None] == np.array(
              list(common_batch_labels))[None, :])[0]
      # reduce the counts of these labels in `batch_cnts` by 1
      delete = []
      for i in ix:
        batch_cnts[i] -= 1
        if (batch_labels[i] > 0) and (batch_cnts[i] <= 2):
          delete.append(i)
        elif (batch_labels[i] < 0) and (batch_cnts[i] < 2):
          delete.append(i)

      for i in sorted(delete, reverse=True):
        del batch_cnts[i]
        del batch_labels[i]

      tensors, network_structure, con_order = _batch_cont(
          t1, t2, tensors, network_structure, con_order, common_batch_labels,
          labels_t1, labels_t2, backend_obj)
    # in all other cases do a regular tensordot
    else:
      # for len(t1_cont)~<20 this is faster than np.argsort
      ind_sort = [t1_cont.index(l) for l in sorted(t1_cont)]
      tensors.append(
          backend_obj.tensordot(
              t1,
              t2,
              axes=(tuple(t1_cont[i] for i in ind_sort),
                    tuple(t2_cont[i] for i in ind_sort))))
      new_labels = [l for l in labels_t1 if l not in common_labels
                   ] + [l for l in labels_t2 if l not in common_labels]
      network_structure.append(new_labels)
      # remove contracted labels from con_order
      con_order = [c for c in con_order if c not in common_labels]

  # perform outer products and remaining batch contractions
  while len(tensors) > 1:
    t2 = tensors.pop()
    t1 = tensors.pop()
    labels_t2 = network_structure.pop()
    labels_t1 = network_structure.pop()
    # check if there are negative batch indices left
    # (have to be collapsed to a single one)
    common_labels, t1_cont, t2_cont = label_intersection(labels_t1, labels_t2)
    common_batch_labels = set(batch_labels).intersection(common_labels)
    if len(common_batch_labels) > 0:
      # collapse all negative batch indices
      tensors, network_structure, con_order = _batch_cont(
          t1, t2, tensors, network_structure, con_order, common_batch_labels,
          labels_t1, labels_t2, backend_obj)
    else:
      tensors.append(backend_obj.outer_product(t1, t2))
      network_structure.append(labels_t1 + labels_t2)

  # if necessary do a final permutation
  if len(network_structure[0]) > 1:
    labels = network_structure[0]
    final_order = tuple(labels.index(l) for l in out_order)
    return backend_obj.transpose(tensors[0], final_order)
  return tensors[0]


def ncon(
    tensors: Sequence[Union[tn_tensor.Tensor, Tensor]],
    network_structure: Sequence[Sequence[Union[str, int]]],
    con_order: Optional[Sequence] = None,
    out_order: Optional[Sequence] = None,
    check_network: bool = True,
    backend: Optional[Union[Text, AbstractBackend]] = None
) -> Union[tn_tensor.Tensor, Tensor]:
  r"""Contracts a list of backend-tensors or  `Tensor`s 
    according to a tensor network 
    specification.

    The network is provided as a list of lists, one for each
    tensor, specifying the labels for the edges connected to that tensor.
    
    Labels can be any numbers or strings. Negative number-type labels
    and string-type labels with a prepended hyphen ('-') are open labels
    and remain uncontracted.

    Positive number-type labels and string-type labels with no prepended 
    hyphen ('-') are closed labels and are contracted.

    Any open label appearing more than once is treated as an open 
    batch label. Any closed label appearing more than once is treated as 
    a closed batch label.

    Upon finishing the contraction, all open batch labels will have been 
    collapsed into a single dimension, and all closed batch labels will 
    have been summed over.

    If `out_order = None`, output labels are ordered according to descending
    number ordering and ascending ASCII ordering, with number labels always 
    appearing before string labels. Example:
    network_structure = [[-1, 1, '-rick', '2',-2], [-2, '2', 1, '-morty']] 
    results in an output order of [-1, -2, '-morty', '-rick'].

    If `out_order` is given, the indices of the resulting tensor will be
    transposed into this order. 
    
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
      Disallowing `0` as an edge label is legacy behaviour, see
      `original NCON implementation`_. 
    
    .. _original NCON implementation:
      https://arxiv.org/abs/1402.0939
    
    Args:
      tensors: List of backend-tensors or `Tensor`s.
      network_structure: List of lists specifying the tensor network structure.
      con_order: List of edge labels specifying the contraction order.
      out_order: List of edge labels specifying the output order.
      check_network: Boolean flag. If `True` check the network.
      backend: String specifying the backend to use. Defaults to
        `tensornetwork.backend_contextmanager.get_default_backend`.

    Returns:
      The result of the contraction: 
        * A backend-tensor: If all elements of `tensors` are backend-tensors.
        * A `Tensor`: If all elements of `tensors` are `Tensor` objects.
    """

  # TODO (mganahl): for certain cases np.einsum is still faster than ncon:
  # - contractions containing batched outer products with small dimensions
  # This should eventually be fixed, but it's not a priority.

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

  are_tensors = [isinstance(t, tn_tensor.Tensor) for t in tensors]
  tensors_set = {t for t in tensors if isinstance(t, tn_tensor.Tensor)}
  if not all(n.backend.name == backend_obj.name for n in tensors_set):
    raise ValueError("Some tensors have backends different from '{}'".format(
        backend_obj.name))

  _tensors = []
  for t in tensors:
    if isinstance(t, tn_tensor.Tensor):
      _tensors.append(t.array)
    else:
      _tensors.append(t)
  _tensors = [backend_obj.convert_to_tensor(t) for t in _tensors]
  if check_network:
    _check_network(network_structure, [t.shape for t in _tensors], con_order,
                   out_order)
  network_structure, mapping = _canonicalize_network_structure(
      network_structure)
  flat_labels = [l for sublist in network_structure for l in sublist]
  unique_flat_labels = list(set(flat_labels))
  if out_order is None:
    # negative batch labels (negative labels appearing more than once)
    # are subject to the same output ordering as regular output labels
    out_order = sorted([l for l in unique_flat_labels if l < 0], reverse=True)
  else:
    out_order = [mapping[o] for o in out_order]
  if con_order is None:
    # canonicalization of network structure takes care of appropriate
    # contraction ordering (i.e. use ASCII ordering for str and
    # regular ordering for int)
    # all positive labels appearing are considered proper contraction labels.
    con_order = sorted([l for l in unique_flat_labels if l > 0])
  else:
    con_order = [mapping[o] for o in con_order]
  if backend not in _CACHED_JITTED_NCONS:
    _CACHED_JITTED_NCONS[backend] = backend_obj.jit(
        _jittable_ncon, static_argnums=(1, 2, 3, 4, 5))
  sizes = tuple(len(l) for l in network_structure)
  res_tensor = _CACHED_JITTED_NCONS[backend](_tensors, tuple(flat_labels),
                                             sizes, tuple(con_order),
                                             tuple(out_order), backend_obj)
  if all(are_tensors):
    return tn_tensor.Tensor(res_tensor, backend=backend_obj)
  return res_tensor

def finalize(ncon_builder: tn_tensor.NconBuilder) -> tn_tensor.Tensor:
  return ncon(
      ncon_builder.tensors, 
      ncon_builder.axes,
      backend=ncon_builder.tensors[0].backend)
