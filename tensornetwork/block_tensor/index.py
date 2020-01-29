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
from tensornetwork.block_tensor.charge import BaseCharge
import copy
from typing import List, Union, Any, Optional, Tuple, Text


class Index:
  """
  An index class to store indices of a symmetric tensor.
  An index keeps track of all its childs by storing references
  to them (i.e. it is a binary tree).
  """

  def __init__(self,
               charges: BaseCharge,
               flow: int,
               name: Optional[Text] = None,
               left_child: Optional["Index"] = None,
               right_child: Optional["Index"] = None):
    self._charges = charges  #ChargeCollection([charges])
    self.flow = flow
    self.left_child = left_child
    self.right_child = right_child
    self.name = name

  def __repr__(self):
    return str(self.dimension)

  @property
  def is_leave(self):
    return (self.left_child is None) and (self.right_child is None)

  @property
  def dimension(self):
    return np.prod([len(i.charges) for i in self.get_elementary_indices()])

  def _copy_helper(self, index: "Index", copied_index: "Index") -> None:
    """
    Helper function for copy
    """
    if index.left_child != None:
      left_copy = Index(
          charges=copy.deepcopy(index.left_child.charges),
          flow=copy.deepcopy(index.left_child.flow),
          name=copy.deepcopy(index.left_child.name))

      copied_index.left_child = left_copy
      self._copy_helper(index.left_child, left_copy)
    if index.right_child != None:
      right_copy = Index(
          charges=copy.deepcopy(index.right_child.charges),
          flow=copy.deepcopy(index.right_child.flow),
          name=copy.deepcopy(index.right_child.name))
      copied_index.right_child = right_copy
      self._copy_helper(index.right_child, right_copy)

  def copy(self):
    """
    Returns:
      Index: A deep copy of `Index`. Note that all children of
        `Index` are copied as well.
    """
    index_copy = Index(
        charges=copy.deepcopy(self._charges),
        flow=copy.deepcopy(self.flow),
        name=self.name)

    self._copy_helper(self, index_copy)
    return index_copy

  def _leave_helper(self, index: "Index", leave_list: List) -> None:
    if index.left_child:
      self._leave_helper(index.left_child, leave_list)
    if index.right_child:
      self._leave_helper(index.right_child, leave_list)
    if (index.left_child is None) and (index.right_child is None):
      leave_list.append(index)

  def get_elementary_indices(self) -> List:
    """
    Returns:
    List: A list containing the elementary indices (the leaves) 
      of `Index`.
    """
    leave_list = []
    self._leave_helper(self, leave_list)
    return leave_list

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
    if self.is_leave:
      return self._charges
    return self.left_child.charges * self.left_child.flow + self.right_child.charges * self.right_child.flow


def fuse_index_pair(left_index: Index,
                    right_index: Index,
                    flow: Optional[int] = False) -> Index:
  """
  Fuse two consecutive indices (legs) of a symmetric tensor.
  Args:
    left_index: A tensor Index.
    right_index: A tensor Index.
    flow: An optional flow of the resulting `Index` object.
  Returns:
    Index: The result of fusing `index1` and `index2`.
  """
  #Fuse the charges of the two indices
  if left_index is right_index:
    raise ValueError(
        "index1 and index2 are the same object. Can only fuse distinct objects")

  return Index(
      charges=None, flow=flow, left_child=left_index, right_child=right_index)


def fuse_indices(indices: List[Index], flow: Optional[int] = False) -> Index:
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
    index = fuse_index_pair(index, indices[n], flow=flow)
  return index


def split_index(index: Index) -> Tuple[Index, Index]:
  """
  Split an index (leg) of a symmetric tensor into two legs.
  Args:
    index: A tensor Index.
  Returns:
    Tuple[Index, Index]: The result of splitting `index`.
  """
  if index.is_leave:
    raise ValueError("cannot split an elementary index")

  return index.left_child, index.right_child
