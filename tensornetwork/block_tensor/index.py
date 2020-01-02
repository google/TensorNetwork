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
from tensornetwork.network_components import Node, contract, contract_between
# pylint: disable=line-too-long
from tensornetwork.backends import backend_factory
import copy
from typing import List, Union, Any, Optional, Tuple, Text


def fuse_charge_pair(q1: Union[List, np.ndarray], flow1: int,
                     q2: Union[List, np.ndarray], flow2: int) -> np.ndarray:
  """
  Fuse charges `q1` with charges `q2` by simple addition (valid
  for U(1) charges). `q1` and `q2` typically belong to two consecutive
  legs of `BlockSparseTensor`.
  Given `q1 = [0,1,2]` and `q2 = [10,100]`, this returns
  `[10, 100, 11, 101, 12, 102]`.
  When using row-major ordering of indices in `BlockSparseTensor`, 
  the position of q1 should be "to the left" of the position of q2.

  Args:
    q1: Iterable of integers
    flow1: Flow direction of charge `q1`.
    q2: Iterable of integers
    flow2: Flow direction of charge `q2`.
  Returns:
    np.ndarray: The result of fusing `q1` with `q2`.
  """
  return np.reshape(
      flow1 * np.asarray(q1)[:, None] + flow2 * np.asarray(q2)[None, :],
      len(q1) * len(q2))


def fuse_charges(charges: List[Union[List, np.ndarray]],
                 flows: List[int]) -> np.ndarray:
  """
  Fuse all `charges` by simple addition (valid
  for U(1) charges). Charges are fused from "right to left", 
  in accordance with row-major order (see `fuse_charges_pair`).

  Args:
    chargs: A list of charges to be fused.
    flows: A list of flows, one for each element in `charges`.
  Returns:
    np.ndarray: The result of fusing `charges`.
  """
  if len(charges) == 1:
    #nothing to do
    return charges[0]
  fused_charges = charges[0] * flows[0]
  for n in range(1, len(charges)):
    fused_charges = fuse_charge_pair(
        q1=fused_charges, flow1=1, q2=charges[n], flow2=flows[n])
  return fused_charges


def fuse_degeneracies(degen1: Union[List, np.ndarray],
                      degen2: Union[List, np.ndarray]) -> np.ndarray:
  """
  Fuse degeneracies `degen1` and `degen2` of two leg-charges 
  by simple kronecker product. `degen1` and `degen2` typically belong to two 
  consecutive legs of `BlockSparseTensor`.
  Given `degen1 = [1, 2, 3]` and `degen2 = [10, 100]`, this returns
  `[10, 100, 20, 200, 30, 300]`.
  When using row-major ordering of indices in `BlockSparseTensor`, 
  the position of `degen1` should be "to the left" of the position of `degen2`.
  Args:
    degen1: Iterable of integers
    degen2: Iterable of integers
  Returns:
    np.ndarray: The result of fusing `dege1` with `degen2`.
  """
  return np.reshape(degen1[:, None] * degen2[None, :],
                    len(degen1) * len(degen2))


def unfuse(fused_indices: np.ndarray, len_left: int,
           len_right: int) -> Tuple[np.ndarray, np.ndarray]:
  """
  Given an np.ndarray `fused_indices` of integers denoting 
  index-positions of elements within a 1d array, `unfuse`
  obtains the index-positions of the elements in the left and 
  right np.ndarrays `left`, `right` which, upon fusion, 
  are placed at the index-positions given by 
  `fused_indices` in the fused np.ndarray.
  An example will help to illuminate this:
  Given np.ndarrays `left`, `right` and the result
  of their fusion (`fused`):

  ```
  left = [0,1,0,2]
  right = [-1,3,-2]    
  fused = fuse_charges([left, right], flows=[1,1]) 
  print(fused) #[-1  3 -2  0  4 -1 -1  3 -2  1  5  0]
  ```

  we want to find which elements in `left` and `right`
  fuse to a value of 0. In the above case, there are two 
  0 in `fused`: one is obtained from fusing `left[1]` and
  `right[0]`, the second one from fusing `left[3]` and `right[2]`
  `unfuse` returns the index-positions of these values within
  `left` and `right`, that is

  ```
  left_index_values, right_index_values = unfuse(np.nonzero(fused==0)[0], len(left), len(right))
  print(left_index_values) # [1,3]
  print(right_index_values) # [0,2]
  ```

  Args:
    fused_indices: A 1d np.ndarray of integers.
    len_left: The length of the left np.ndarray.
    len_right: The length of the right np.ndarray.
  Returns:
    (np.ndarry, np.ndarray)
  """
  right = np.mod(fused_indices, len_right)
  left = np.floor_divide(fused_indices - right, len_right)
  return left, right


def _copy_charges(charges):
  cs = []
  for n in range(len(charges)):
    c = type(charges[n]).__new__(type(
        charges[n]))  #create a new charge object of type type(other)
    c.__init__(charges[n].charges.copy())
    cs.append(c)
  return cs


class BaseCharge:
  """
  Base class for fundamental charges (i.e. for symmetries that 
  are not products of smaller groups)
  """

  def __init__(self, charges: np.ndarray) -> None:
    if not isinstance(charges, np.ndarray):
      raise TypeError("only np.ndarray allowed for argument `charges` "
                      "in BaseCharge.__init__(charges)")

    self.charges = np.asarray(charges)

  def __add__(self, other: "BaseCharge") -> "BaseCharge":
    raise NotImplementedError("`__add__` is not implemented for `BaseCharge`")

  def __mul__(self, number: int) -> "BaseCharge":
    raise NotImplementedError("`__mul__` is not implemented for `BaseCharge`")

  def __rmul__(self, number: int) -> "BaseCharge":
    raise NotImplementedError("`__rmul__` is not implemented for `BaseCharge`")

  def __matmul__(self, other: "BaseCharge") -> "Charge":
    raise NotImplementedError(
        "`__matmul__` is not implemented for `BaseCharge`")

  def __len__(self) -> int:
    return len(self.charges)

  def __repr__(self) -> str:
    return self.charges.__repr__()

  @property
  def dual_charges(self) -> np.ndarray:
    raise NotImplementedError(
        "`dual_charges` is not implemented for `BaseCharge`")

  def get_charges(self, dual: bool) -> np.ndarray:
    if dual:
      return self.dual_charges
    return self.charges


class U1ChargeCoerced:
  """
  A simple charge class for a single U1 symmetry.
  """

  def __init__(self,
               charges: List[np.ndarray],
               offsets: Optional[np.ndarray] = None,
               shifts: Optional[np.ndarray] = None) -> None:
    itemsizes = [8 * c.dtype.itemsize for c in charges]
    if np.sum(itemsizes) > 64:
      raise TypeError("number of bits required to store all charges "
                      "in a single int is larger than 64")
    if np.sum(itemsizes) == 16:
      dtype = np.int16
    if np.sum(itemsizes) > 16:
      dtype = np.int32
    if np.sum(itemsizes) > 32:
      dtype = np.int64

    if shifts is None:
      self.shifts = np.flip(np.append(0, np.cumsum(np.flip(
          itemsizes[1::])))).astype(dtype)
    else:
      self.shifts = shifts

    dtype_charges = [c.astype(dtype) for c in charges]
    if offsets is None:
      offsets = [np.min(dtype_charges[n]) for n in range(len(dtype_charges))]
      pos_charges = [
          dtype_charges[n] - offsets[n] for n in range(len(dtype_charges))
      ]
      self.offsets = np.sum([
          np.left_shift(offsets[n], self.shifts[n])
          for n in range(len(dtype_charges))
      ],
                            axis=0).astype(dtype)
      self._charges = np.sum([
          np.left_shift(pos_charges[n], self.shifts[n])
          for n in range(len(dtype_charges))
      ],
                             axis=0).astype(dtype)
    else:
      if len(charges) > 1:
        raise ValueError(
            'if offsets is given, only a single charge array can be passed')
      self.offsets = offsets
      self._charges = dtype_charges[0]

  @property
  def num_symmetries(self):
    return len(self.shifts)

  def __add__(self, other: "U1ChargeCoerced") -> "U1ChargeCoerced":
    """
    Fuse the charges of `self` with charges of `other`, and 
    return a new `U1Charge` object holding the result.
    Args: 
      other: A `U1ChargeCoerced` object.
    Returns:
      U1ChargeCoerced: The result of fusing `self` with `other`.
    """
    if self.num_symmetries != other.num_symmetries:
      raise ValueError(
          "cannot fuse charges with different number of symmetries")

    if not np.all(self.shifts == other.shifts):
      raise ValueError(
          "Cannot fuse U1-charges with different shifts {} and {}".format(
              self.shifts, other.shifts))
    offsets = np.sum([self.offsets, other.offsets])
    fused = np.reshape(self._charges[:, None] + other.charges[None, :],
                       len(self._charges) * len(other.charges))
    return U1ChargeCoerced(charges=[fused], offsets=offsets, shifts=self.shifts)

  def __repr__(self):
    return 'U1-charge: \n' + 'shifts: ' + self.shifts.__repr__(
    ) + '\n' + 'offsets: ' + self.offsets.__repr__(
    ) + '\n' + 'charges: ' + self._charges.__repr__()

  # def __matmul__(self, other: Union["U1Charge", "Charge"]) -> "Charge":
  #   c1 = U1Charge(self._charges.copy())  #make a copy of the charges (np.ndarray)
  #   if isinstance(other, U1Charge):
  #     c2 = type(other).__new__(
  #         type(other))  #create a new charge object of type type(other)
  #     c2.__init__(other.charges.copy())
  #     return Charge([c1, c2])
  #   #`other` should be of type `Charge`.
  #   return Charge([c1] + _copy_charges(other.charges))

  @property
  def dual_charges(self) -> np.ndarray:
    #the dual of a U1 charge is its negative value
    return (self._charges + self.offsets) * self._charges.dtype.type(-1)

  @property
  def charges(self) -> np.ndarray:
    return self._charges + self.offsets

  def get_charges(self, dual: bool) -> np.ndarray:
    if dual:
      return self.dual_charges
    return self._charges + self.offsets

  def nonzero(self, target_charges: Union[List, np.ndarray]) -> np.ndarray:
    if len(target_charges) != len(self.shifts):
      raise ValueError("len(target_charges) = {} is different "
                       "from len(U1ChargeCoerced.shifts) = {}".format(
                           len(target_charges), len(self.shifts)))
    charge = np.asarray(target_charges).astype(self._charges.dtype)
    target = np.sum([
        np.left_shift(charge[n], self.shifts[n])
        for n in range(len(self.shifts))
    ])
    return np.nonzero(self._charges + self.offsets == target)[0]


class U1Charge(BaseCharge):
  """
  A simple charge class for a single U1 symmetry.
  """

  def __init__(self, charges: np.ndarray) -> None:
    super().__init__(charges)

  def __add__(self, other: "U1Charge") -> "U1Charge":
    """
    Fuse the charges of `self` with charges of `other`, and 
    return a new `U1Charge` object holding the result.
    Args: 
      other: A `U1Charge` object.
    Returns:
      U1Charge: The result of fusing `self` with `other`.
    """
    fused = np.reshape(self.charges[:, None] + other.charges[None, :],
                       len(self.charges) * len(other.charges))

    return U1Charge(charges=fused)

  def __mul__(self, number: int) -> "U1Charge":
    return U1Charge(charges=self.charges * number)

  def __rmul__(self, number: int) -> "U1Charge":
    return U1Charge(charges=self.charges * number)

  def __matmul__(self, other: Union["U1Charge", "Charge"]) -> "Charge":
    c1 = U1Charge(self.charges.copy())  #make a copy of the charges (np.ndarray)
    if isinstance(other, U1Charge):
      c2 = type(other).__new__(
          type(other))  #create a new charge object of type type(other)
      c2.__init__(other.charges.copy())
      return Charge([c1, c2])
    #`other` should be of type `Charge`.
    return Charge([c1] + _copy_charges(other.charges))

  @property
  def dual_charges(self):
    #the dual of a U1 charge is its negative value
    return self.charges * self.charges.dtype.type(-1)


class Z2Charge(BaseCharge):
  """
  A simple charge class for a single Z2 symmetry.
  """

  def __init__(self, charges: np.ndarray) -> None:
    if charges.dtype is not np.dtype(np.bool):
      raise TypeError("Z2 charges have to be boolian")
    super().__init__(charges)

  def __add__(self, other: "U1Charge") -> "U1Charge":
    """
    Fuse the charges of `self` with charges of `other`, and 
    return a new `U1Charge` object holding the result.
    Args: 
      other: A `U1Charge` object.
    Returns:
      U1Charge: The result of fusing `self` with `other`.
    """
    fused = np.reshape(
        np.logical_xor(self.charges[:, None], other.charges[None, :]),
        len(self.charges) * len(other.charges))

    return U1Charge(charges=fused)

  def __mul__(self, number: int) -> "U1Charge":
    return U1Charge(charges=self.charges * number)

  def __rmul__(self, number: int) -> "U1Charge":
    return U1Charge(charges=self.charges * number)

  def __matmul__(self, other: Union["U1Charge", "Charge"]) -> "Charge":
    c1 = U1Charge(self.charges.copy())  #make a copy of the charges (np.ndarray)
    if isinstance(other, U1Charge):
      c2 = type(other).__new__(
          type(other))  #create a new charge object of type type(other)
      c2.__init__(other.charges.copy())
      return Charge([c1, c2])
    #`other` should be of type `Charge`.
    return Charge([c1] + _copy_charges(other.charges))

  @property
  def dual_charges(self):
    #Z2 charges are self-dual
    return self.charges


class Charge:

  def __init__(self, charges: List[Union[np.ndarray, BaseCharge]]) -> None:
    if not isinstance(charges, list):
      raise TypeError("only list allowed for argument `charges` "
                      "in BaseCharge.__init__(charges)")
    if not np.all([len(c) == len(charges[0]) for c in charges]):
      raise ValueError("not all charges have the same length. "
                       "Got lengths = {}".format([len(c) for c in charges]))
    for n in range(len(charges)):
      if not isinstance(charges[n], BaseCharge):
        raise TypeError(
            "`Charge` can only be initialized with a list of `BaseCharge`. Found {} instead"
            .format(type(charges[n])))

    self.charges = charges

  def __add__(self, other: "Charge") -> "Charge":
    """
    Fuse `self` with `other`.
    Args:
      other: A `Charge` object.
    Returns:
      Charge: The result of fusing `self` with `other`.
    """
    return Charge([c1 + c2 for c1, c2 in zip(self.charges, other.charges)])

  def __matmul__(self, other: Union["Charge", BaseCharge]) -> "Charge":
    """
    Product of `self` with `other` (group product).
    Args:
      other: A `BaseCharge` or `Charge` object.
    Returns:
      Charge: The resulting charge of the product of `self` with `other`.

    """
    if isinstance(other, BaseCharge):
      c = type(other).__new__(
          type(other))  #create a new charge object of type type(other)
      c.__init__(other.charges.copy())
      return Charge(self.charges + [c])
    elif isinstance(other, Charge):
      return Charge(_copy_charges(self.charges) + _copy_charges(other.charges))

    raise TypeError("datatype not understood")

  def __mul__(self, number: int) -> "Charge":
    return Charge(charges=[c * number for c in self.charges])

  def __rmul__(self, number: int) -> "Charge":
    return Charge(charges=[c * number for c in self.charges])

  def __len__(self):
    return len(self.charges[0])

  def __repr__(self):
    return self.charges.__repr__()

  def get_charges(self, dual: bool) -> List[np.ndarray]:
    return [c.get_charges(dual) for c in self.charges]


class Index:
  """
  An index class to store indices of a symmetric tensor.
  An index keeps track of all its childs by storing references
  to them (i.e. it is a binary tree).
  """

  def __init__(self,
               charges: Union[Charge, BaseCharge],
               flow: int,
               name: Optional[Text] = None,
               left_child: Optional["Index"] = None,
               right_child: Optional["Index"] = None):
    if isinstance(charges, BaseCharge):
      self._charges = Charge([charges])
    elif isinstance(charges, Charge):
      self._charges = charges
    self.flow = flow
    self.left_child = left_child
    self.right_child = right_child
    self._name = name

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
          charges=copy.copy(index.left_child.charges),
          flow=copy.copy(index.left_child.flow),
          name=copy.copy(index.left_child.name))

      copied_index.left_child = left_copy
      self._copy_helper(index.left_child, left_copy)
    if index.right_child != None:
      right_copy = Index(
          charges=copy.copy(index.right_child.charges),
          flow=copy.copy(index.right_child.flow),
          name=copy.copy(index.right_child.name))
      copied_index.right_child = right_copy
      self._copy_helper(index.right_child, right_copy)

  def copy(self):
    """
    Returns:
      Index: A deep copy of `Index`. Note that all children of
        `Index` are copied as well.
    """
    index_copy = Index(
        charges=copy.copy(self._charges),
        flow=copy.copy(self.flow),
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

  @property
  def name(self):
    if self._name:
      return self._name
    if self.is_leave:
      return self.name
    return self.left_child.name + ' & ' + self.right_child.name


def fuse_index_pair(left_index: Index,
                    right_index: Index,
                    flow: Optional[int] = 1) -> Index:
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


def fuse_indices(indices: List[Index], flow: Optional[int] = 1) -> Index:
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
