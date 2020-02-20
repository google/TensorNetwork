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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensornetwork.block_sparse.charge import BaseCharge, fuse_charges
import copy
from typing import List, Union, Any, Optional, Tuple, Text


class Index:
  """
  An index class to store indices of a symmetric tensor.
  """

  def __init__(self, charges: Union[List[BaseCharge], BaseCharge],
               flow: Union[List[bool], bool]) -> None:
    """
    Initialize an `Index` object.
    """
    if isinstance(charges, BaseCharge):
      charges = [charges]
    self._charges = charges
    if np.isscalar(flow):
      flow = [flow]
    if not all([isinstance(f, (np.bool_, np.bool, bool)) for f in flow]):
      raise TypeError("flows have to be boolean. Found flow = {}".format(flow))
    self.flow = flow

  def __len__(self):
    return self.dim

  def __repr__(self):
    return str(self.dim)

  @property
  def dim(self):
    return np.prod([i.dim for i in self._charges])

  def __eq__(self, other):
    if len(other._charges) != len(self._charges):
      return False
    for n in range(len(self._charges)):
      if not np.all(
          self._charges[n].unique_charges == other._charges[n].unique_charges):
        return False
      if not np.all(
          self._charges[n].charge_labels == other._charges[n].charge_labels):
        return False
    if not np.all(np.asarray(self.flow) == np.asarray(other.flow)):
      return False
    return True

  def copy(self):
    """
    Returns:
      Index: A deep copy of `Index`. Note that all children of
        `Index` are copied as well.
    """
    index_copy = Index(
        charges=[c.copy() for c in self._charges],
        flow=copy.deepcopy(self.flow))

    return index_copy

  @property
  def flat_charges(self) -> List:
    """
    Returns:
    List: A list containing the elementary indices (the leaves) 
      of `Index`.
    """
    return self._charges

  @property
  def flat_flows(self) -> List:
    """
    Returns:
    List: A list containing the elementary indices (the leaves) 
      of `Index`.
    """
    return list(self.flow)

  def flip_flow(self):
    self.flow = np.logical_not(self.flow)
    return self

  def __mul__(self, index: "Index") -> "Index":
    """
    Merge `index` and self into a single larger index.
    The flow of the resulting index is set to 1.
    Flows of `self` and `index` are multiplied into 
    the charges upon fusing.n
    """
    return fuse_index_pair(self, index)

  @property
  def charges(self):
    return fuse_charges(self.flat_charges, self.flat_flows)


def fuse_index_pair(left_index: Index, right_index: Index) -> Index:
  """
  Fuse two consecutive indices (legs) of a symmetric tensor.
  Args:
    left_index: A tensor Index.
    right_index: A tensor Index.
    flow: An optional flow of the resulting `Index` object.
  Returns:
    Index: The result of fusing `index1` and `index2`.
  """

  return Index(
      charges=left_index.flat_charges + right_index.flat_charges,
      flow=left_index.flat_flows + right_index.flat_flows)


def fuse_indices(indices: List[Index]) -> Index:
  """
  Fuse a list of indices (legs) of a symmetric tensor.
  Args:
    indices: A list of tensor Index objects
    flow: An optional flow of the resulting `Index` object.
  Returns:
    Index: The result of fusing `indices`.
  """

  index = indices[0]
  for n in range(1, len(indices)):
    index = fuse_index_pair(index, indices[n])
  return index
