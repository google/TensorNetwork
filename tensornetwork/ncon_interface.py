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
from typing import Any, Sequence, List, Optional, Union, Text, Tuple, Dict
from tensornetwork import network_components
from tensornetwork.backend_contextmanager import get_default_backend
from tensornetwork.backends import backend_factory
from tensornetwork.backends.abstract_backend import AbstractBackend
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
  int_labels = [o for o in flat_labels if not isinstance(o, str)]
  str_labels = [o for o in flat_labels if isinstance(o, str)]

  arr = np.array(int_labels)
  int_out_labels = list(np.unique(arr[arr < 0]).astype(int))
  int_cont_labels = []

  for label in int_labels:
    if (label >= 0) and (label not in int_cont_labels):
      int_cont_labels.append(label)

  str_out_labels = []
  str_cont_labels = []
  for label in str_labels:
    if (label[0] == '-') and (label not in str_out_labels):
      str_out_labels.append(label)
    if (label[0] != '-') and (label not in str_cont_labels):
      str_cont_labels.append(label)

  int_cont_labels = sorted(int_cont_labels)
  # pylint: disable=unnecessary-lambda
  str_cont_labels = sorted(str_cont_labels, key=lambda x: str(x))

  int_out_labels = sorted(int_out_labels)[::-1]
  # pylint: disable=unnecessary-lambda
  str_out_labels = sorted(str_out_labels, key=lambda x: str(x))
  # pylint: disable=line-too-long
  return int_cont_labels, str_cont_labels, int_out_labels, str_out_labels


def _canonicalize_network_structure(
    network_structure: Sequence[Sequence[Union[int, str]]]
) -> Tuple[List[np.ndarray], Dict]:
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
  neg_int_labels = [l for l in flat_labels if not isinstance(l, str) and l < 0]
  pos_int_labels = [l for l in flat_labels if not isinstance(l, str) and l > 0]
  neg_str_labels = [
      l for l in flat_labels if isinstance(l, str) and l[0] == '-'
  ]
  pos_str_labels = [
      l for l in flat_labels if isinstance(l, str) and l[0] != '-'
  ]
  neg_str_labels = list(np.unique(neg_str_labels))[::-1]
  neg_int_labels = list(np.unique(neg_int_labels))

  pos_str_labels = list(np.unique(pos_str_labels))
  pos_int_labels = list(np.unique(pos_int_labels))
  neg_mapping = dict(
      zip(neg_str_labels + neg_int_labels,
          np.arange(-len(neg_int_labels + neg_str_labels), 0)))
  pos_mapping = dict(
      zip(pos_int_labels + pos_str_labels,
          np.arange(1, 1 + len(pos_int_labels + pos_str_labels))))
  neg_mapping.update(pos_mapping)
  mapped_network_structure = [
      np.asarray([neg_mapping[label]
                  for label in labels])
      for labels in network_structure
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
    raise ValueError(f"only alphanumeric values allowed for string labels, "
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
  locations = {}
  for l in cont_labels:
    boolean_mask = [[l1 == l for l1 in labels] for labels in network_structure]
    locations[l] = np.nonzero(boolean_mask)[0]

  mismatched_labels = []
  for label, locs in locations.items():
    inds = [
        np.nonzero([l1 == label
                    for l1 in network_structure[loc]])[0]
        for loc in locs
    ]
    if len(inds) > 0:
      label_dims = np.concatenate([
          np.array(tensor_dimensions[loc])[inds[n]]
          for n, loc in enumerate(locs)
      ])
      if not np.all(label_dims == label_dims[0]):
        mismatched_labels.append(label)
  if len(mismatched_labels) > 0:
    raise ValueError(
        f"tensor dimensions for labels {mismatched_labels} are mismatching")


def _partial_trace(
    tensor: Tensor, labels: np.ndarray,
    backend_obj: AbstractBackend) -> Tuple[Tensor, np.ndarray, List]:
  """
  Perform the partial trace of `tensor`.
  All labels appearing twice in `labels` are traced out.
  Args:
    tensor: A tensor.
    labels: The ncon-style labels of `tensor`.
  Returns:
    Tensor: The result of the tracing.
  """
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


def _batch_cont(
    t1: Tensor, t2: Tensor, tensors: List[Tensor],
    network_structure: List[np.ndarray], con_order: np.ndarray,
    common_batch_labels: np.ndarray, labels_t1: np.ndarray,
    labels_t2: np.ndarray, backend_obj: AbstractBackend
) -> Tuple[Tensor, List[np.ndarray], np.ndarray]:
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
    List[np.ndarray]: Updated `network_structure`.
    np.ndarray: Updated `con_order` (contraction order).
  """
  #find positions of common batch labels
  _, _, t1_batch_pos = np.intersect1d(
      common_batch_labels, labels_t1, assume_unique=True, return_indices=True)
  _, _, t2_batch_pos = np.intersect1d(
      common_batch_labels, labels_t2, assume_unique=True, return_indices=True)

  #find positions of contracted non-batch labels
  non_batch_labels_t1 = labels_t1[np.logical_not(
      np.isin(labels_t1, common_batch_labels))]
  non_batch_labels_t2 = labels_t2[np.logical_not(
      np.isin(labels_t2, common_batch_labels))]
  common_contracted_labels = np.intersect1d(
      non_batch_labels_t1, non_batch_labels_t2, assume_unique=True)
  _, _, t1_cont = np.intersect1d(
      common_contracted_labels,
      labels_t1,
      assume_unique=True,
      return_indices=True)
  _, _, t2_cont = np.intersect1d(
      common_contracted_labels,
      labels_t2,
      assume_unique=True,
      return_indices=True)

  # find positions of uncontracted non-batch labels
  free_pos_t1 = np.setdiff1d(
      np.arange(len(labels_t1)), np.append(t1_cont, t1_batch_pos))
  free_pos_t2 = np.setdiff1d(
      np.arange(len(labels_t2)), np.append(t2_cont, t2_batch_pos))

  t1_shape = np.array(backend_obj.shape_tuple(t1))
  t2_shape = np.array(backend_obj.shape_tuple(t2))

  newshape_t1 = (np.prod(t1_shape[t1_batch_pos]),
                 np.prod(t1_shape[free_pos_t1]), np.prod(t1_shape[t1_cont]))
  newshape_t2 = (np.prod(t2_shape[t2_batch_pos]), np.prod(t2_shape[t2_cont]),
                 np.prod(t2_shape[free_pos_t2]))

  #bring batch labels to the front
  order_t1 = tuple(np.concatenate([t1_batch_pos, free_pos_t1, t1_cont]))
  order_t2 = tuple(np.concatenate([t2_batch_pos, t2_cont, free_pos_t2]))

  mat1 = backend_obj.reshape(backend_obj.transpose(t1, order_t1), newshape_t1)
  mat2 = backend_obj.reshape(backend_obj.transpose(t2, order_t2), newshape_t2)
  result = backend_obj.matmul(mat1, mat2)
  final_shape = tuple(
      np.concatenate([
          t1_shape[t1_batch_pos], t1_shape[free_pos_t1], t2_shape[free_pos_t2]
      ]))
  result = backend_obj.reshape(result, final_shape)

  # update labels, tensors, network_structure and con_order
  new_labels = np.concatenate(
      [labels_t1[t1_batch_pos], labels_t1[free_pos_t1], labels_t2[free_pos_t2]])
  network_structure.append(new_labels)
  tensors.append(result)
  if len(con_order) > 0:
    con_order = np.delete(
        con_order,
        np.intersect1d(
            common_contracted_labels,
            con_order,
            assume_unique=True,
            return_indices=True)[2])

  return tensors, network_structure, con_order


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
  slices = np.append(0, np.cumsum(sizes))
  network_structure = [
      np.array(flat_labels)[slices[n]:slices[n + 1]]
      for n in range(len(slices) - 1)
  ]
  con_order = np.array(con_order)
  out_order = np.array(out_order)
  # pylint: disable=unnecessary-comprehension
  init_con_order = [c for c in con_order]
  init_network_structure = [list(c) for c in network_structure]
  # partial trace
  for n, tensor in enumerate(tensors):
    tensors[n], network_structure[n], contracted_labels = _partial_trace(
        tensor, network_structure[n], backend_obj)
    con_order = np.delete(
        con_order,
        np.intersect1d(con_order, contracted_labels, return_indices=True)[1])

  # contracted all positive labels appearing only once in `network_structure`
  unique_labels, label_cnts = np.unique(
      np.concatenate(network_structure), return_counts=True)
  contractable_labels = unique_labels[np.logical_and(label_cnts == 1,
                                                     unique_labels > 0)]
  locs = np.sort(
      np.nonzero([
          np.any(np.isin(labels, contractable_labels))
          for labels in network_structure
      ])[0])

  for loc in locs:
    labels = network_structure[loc]
    contractable_inds = np.nonzero(np.isin(labels, contractable_labels))[0]
    network_structure[loc] = np.delete(labels, contractable_inds)
    tensors[loc] = backend_obj.sum(tensors[loc], tuple(contractable_inds))

  # update con_order after collapsing single labels
  con_order = np.delete(con_order,
                        np.nonzero(np.isin(con_order, contractable_labels))[0])

  # perform binary and batch contractions
  skip_counter = 0
  while len(con_order) > 0:
    unique_labels, label_cnts = np.unique(
        np.concatenate(network_structure), return_counts=True)
    batch_labels = unique_labels[np.logical_or(
        label_cnts > 2, np.logical_and(label_cnts == 2, unique_labels < 0))]

    # the next index to be contracted
    cont_ind = con_order[0]
    if cont_ind in batch_labels:
      # if its still a batch index then do it later
      con_order = np.append(np.delete(con_order, 0), cont_ind)
      skip_counter += 1
      # avoid being stuck in an infinite loop
      if skip_counter > len(con_order):
        raise ValueError(f"ncon seems stuck in an infinite loop. \n"
                         f"Please check if `con_order` = {init_con_order} is "
                         f"a valid contraction order for \n"
                         f"`network_structure` = {init_network_structure}")
      continue

    # find locations of `cont_ind` in `network_structure`
    locs = np.sort(
        np.nonzero([np.isin(cont_ind, labels) for labels in network_structure
                   ])[0])
    t2 = tensors.pop(locs[1])
    t1 = tensors.pop(locs[0])
    labels_t2 = network_structure.pop(locs[1])
    labels_t1 = network_structure.pop(locs[0])

    common_labels, t1_cont, t2_cont = np.intersect1d(
        labels_t1, labels_t2, assume_unique=True, return_indices=True)
    # check if there are batch labels (i.e. labels appearing more than twice
    # in `network_structure`).
    common_batch_labels = np.intersect1d(
        batch_labels, common_labels, assume_unique=True)
    if common_batch_labels.shape[0] > 0:
      # case1: both tensors have one or more common batch indices -> use matmul
      tensors, network_structure, con_order = _batch_cont(
          t1, t2, tensors, network_structure, con_order, common_batch_labels,
          labels_t1, labels_t2, backend_obj)
    # in all other cases do a regular tensordot
    else:
      ind_sort = np.argsort(t1_cont)
      tensors.append(
          backend_obj.tensordot(
              t1, t2,
              axes=(tuple(t1_cont[ind_sort]), tuple(t2_cont[ind_sort]))))
      network_structure.append(
          np.append(
              np.delete(labels_t1, t1_cont), np.delete(labels_t2, t2_cont)))

      # remove contracted labels from con_order
      con_order = np.delete(
          con_order,
          np.intersect1d(
              con_order, common_labels, assume_unique=True,
              return_indices=True)[1])

  # perform outer products and remaining batch contractions
  while len(tensors) > 1:
    unique_labels, label_cnts = np.unique(
        np.concatenate(network_structure), return_counts=True)
    batch_labels = unique_labels[np.logical_or(
        label_cnts > 2, np.logical_and(label_cnts == 2, unique_labels < 0))]

    t2 = tensors.pop()
    t1 = tensors.pop()
    labels_t2 = network_structure.pop()
    labels_t1 = network_structure.pop()
    # check if there are negative batch indices left
    # (have to be collapsed to a single one)
    common_labels, t1_cont, t2_cont = np.intersect1d(
        labels_t1, labels_t2, assume_unique=True, return_indices=True)
    common_batch_labels = np.intersect1d(
        batch_labels, common_labels, assume_unique=True)
    if common_batch_labels.shape[0] > 0:
      # collapse all negative batch indices
      tensors, network_structure, con_order = _batch_cont(
          t1, t2, tensors, network_structure, con_order, common_batch_labels,
          labels_t1, labels_t2, backend_obj)
    else:
      tensors.append(backend_obj.outer_product(t1, t2))
      network_structure.append(np.append(labels_t1, labels_t2))

  # if necessary do a final permutation
  if len(network_structure[0]) > 0:
    i1, i2 = np.nonzero(out_order[:, None] == network_structure[0][None, :])
    return backend_obj.transpose(tensors[0], tuple(i1[i2]))
  return tensors[0]


def ncon(
    tensors: Sequence[Union[network_components.AbstractNode, Tensor]],
    network_structure: Sequence[Sequence[int]],
    con_order: Optional[Sequence] = None,
    out_order: Optional[Sequence] = None,
    check_network: bool = True,
    backend: Optional[Union[Text, AbstractBackend]] = None
) -> Union[network_components.AbstractNode, Tensor]:
  r"""Contracts a list of tensors or nodes according to a tensor network 
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
    _check_network(network_structure, [t.shape for t in _tensors], con_order,
                   out_order)

  network_structure, mapping = _canonicalize_network_structure(
      network_structure)
  flat_labels = np.concatenate(network_structure)
  if out_order is None:
    # negative batch labels (negative labels appearing more than once)
    # are subject to the same output ordering as regular output labels
    out_order = np.unique(flat_labels[flat_labels < 0])[::-1]
  else:
    out_order = np.asarray([mapping[o] for o in out_order])
  if con_order is None:
    # canonicalization of network structure takes care of appropriate
    # contraction ordering (i.e. use ASCII ordering for str and
    # regular ordering for int)
    # all positive labels appearing are considered proper contraction labels.
    con_order = np.unique(flat_labels[flat_labels > 0])
  else:
    con_order = np.asarray([mapping[o] for o in con_order])
  if backend not in _CACHED_JITTED_NCONS:
    _CACHED_JITTED_NCONS[backend] = backend_obj.jit(
        _jittable_ncon, static_argnums=(1, 2, 3, 4, 5))
  sizes = tuple([len(l) for l in network_structure])
  res_tensor = _CACHED_JITTED_NCONS[backend](_tensors, tuple(flat_labels),
                                             sizes, tuple(con_order),
                                             tuple(out_order), backend_obj)
  if all(are_nodes):
    return network_components.Node(res_tensor, backend=backend_obj)
  return res_tensor
