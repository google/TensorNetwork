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


import numpy as np
import copy
from functools import reduce
from operator import mul

from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.blocksparse_utils import (
    _find_transposed_diagonal_sparse_blocks, _find_diagonal_sparse_blocks,
    get_flat_meta_data, compute_num_nonzero, reduce_charges, _data_initializer)
from tensornetwork.block_sparse.utils import (flatten, _find_best_partition,
                                              intersect, _random, _randn)
from tensornetwork.block_sparse.charge import (fuse_charges, BaseCharge,
                                               charge_equal)
from typing import List, Union, Any, Tuple, Type, Optional, Sequence
Tensor = Any




class ChargeArray:
  """
  Base class for BlockSparseTensor.
  Stores a dense tensor together with its charge data.
  Attributes:
  * _charges: A list of `BaseCharge` objects, one for each leg of the tensor.
  * _flows: An np.ndarray of boolean dtype, storing the flow direction of each 
      leg.
  * data: A flat np.ndarray storing the actual tensor data.
  * _order: A list of list, storing information on how tensor legs are 
    transposed.
  """

  #pylint: disable=unused-argument
  def __init__(self,
               data: np.ndarray,
               charges: List[BaseCharge],
               flows: Union[np.ndarray, List[bool]],
               order: Optional[List[List[int]]] = None,
               check_consistency: Optional[bool] = False) -> None:
    """
    Initialize a `ChargeArray` object. `len(data)` has to 
    be equal to `np.prod([c.dim for c in charges])`.
    
    Args: 
      data: An np.ndarray of the data. 
      charges: A list of `BaseCharge` objects.
      flows: The flows of the tensor indices, `False` for inflowing, `True`
        for outflowing.
      order: An optional order argument, determining the shape and order of the
        tensor.
      check_consistency: Perform error checks at initialization.
    """
    self._charges = charges
    self._flows = np.asarray(flows)

    self.data = np.asarray(data.flat)  #no copy

    if order is None:
      self._order = [[n] for n in range(len(self._charges))]
    else:
      if check_consistency:
        flat_order = []
        for o in order:
          flat_order.extend(o)
        if not np.array_equal(
            np.sort(flat_order), np.arange(len(self._charges))):
          raise ValueError("flat_order = {} is not a permutation of {}".format(
              flat_order, np.arange(len(self._charges))))

      self._order = order

  @classmethod
  def random(cls,
             indices: Union[Tuple[Index], List[Index]],
             boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
             dtype: Optional[Type[np.number]] = None) -> "ChargeArray":
    """
    Initialize a random ChargeArray object with data from a random 
    uniform distribution.
    Args:
      indices: List of `Index` objects.
      boundaries: Tuple of interval boundaries for the random uniform 
        distribution.
      dtype: An optional numpy dtype. The dtype of the ChargeArray
    Returns:
      ChargeArray
    """


    data, charges, flows, order = _data_initializer(
        _random, lambda charges, flows: np.prod([c.dim for c in charges]),
        indices, dtype=dtype)
    return cls(data=data, charges=charges, flows=flows, order=order)

  @property
  def ndim(self) -> int:
    """
    The number of tensor dimensions.
    """
    return len(self._order)

  @property
  def dtype(self) -> Type[np.number]:
    """
    The dtype of `ChargeArray`.
    """
    return self.data.dtype

  @property
  def shape(self) -> Tuple:
    """
    The dense shape of the tensor.
    Returns:
      Tuple: A tuple of `int`.
    """
    return tuple(
        [reduce(mul, [self._charges[n].dim for n in s]) for s in self._order])

  @property
  def size(self) -> int:
    """
    The dense size of the tensor, i.e. the total number of elements, including 
    those which are zero by conservation of charge.
    Returns:
      int: The total number of elements.
    """
    return reduce(mul, [self._charges[n].dim for s in self._order for n in s])

  @property
  def charges(self) -> List[List[BaseCharge]]:
    """
    A list of list of `BaseCharge`.
    The charges, in the current shape and index order as determined 
    by `ChargeArray._order`.
    Returns:
      List of List of BaseCharge
    """
    return [[self._charges[n] for n in o] for o in self._order]

  @property
  def flows(self) -> List[List]:
    """
    A list of list of `bool`.
    The flows, in the current shape and index order as determined by 
    `ChargeArray._order`.
    Returns:
      List of List of bool
    """

    return [[self._flows[n] for n in o] for o in self._order]

  @property
  def flat_charges(self) -> List[BaseCharge]:
    return list([self._charges[o] for o in self.flat_order])

  @property
  def flat_flows(self) -> List:
    return list([self._flows[o] for o in self.flat_order])

  @property
  def flat_order(self) -> np.ndarray:
    """
    The flattened `ChargeArray._oder`.
    """
    return flatten(self._order)

  @property
  def sparse_shape(self) -> Tuple:
    """
    The sparse shape of the tensor.
    Returns:
      Tuple: A tuple of `Index` objects.
    """

    indices = []
    for s in self._order:
      indices.append(
          Index([self._charges[n] for n in s], [self._flows[n] for n in s]))

    return tuple(indices)

  def todense(self) -> np.ndarray:
    """
    Map the sparse tensor to dense storage.
    
    """
    tmp = self.contiguous()
    return np.reshape(tmp.data, tmp.shape)

  def reshape(self, shape: Sequence[Union[Index, int]]) -> "ChargeArray":
    """
    Reshape `tensor` into `shape.
    `ChargeArray.reshape` works the same as the dense 
    version, with the notable exception that the tensor can only be 
    reshaped into a form compatible with its elementary shape. 
    The elementary shape is the shape determined by ChargeArray._charges.
    For example, while the following reshaping is possible for regular 
    dense numpy tensor,
    ```
    A = np.random.rand(6,6,6)
    np.reshape(A, (2,3,6,6))
    ```
    the same code for ChargeArray
    ```
    q1 = U1Charge(np.random.randint(0,10,6))
    q2 = U1Charge(np.random.randint(0,10,6))
    q3 = U1Charge(np.random.randint(0,10,6))
    i1 = Index(charges=q1,flow=False)
    i2 = Index(charges=q2,flow=True)
    i3 = Index(charges=q3,flow=False)
    A=ChargeArray.randn(indices=[i1,i2,i3])
    print(A.shape) #prints (6,6,6)
    A.reshape((2,3,6,6)) #raises ValueError
    ```
    raises a `ValueError` since (2,3,6,6)
    is incompatible with the elementary shape (6,6,6) of the tensor.
    
    Args:
      tensor: A symmetric tensor.
      shape: The new shape. Can either be a list of `Index` 
        or a list of `int`.
    Returns:
      ChargeArray: A new tensor reshaped into `shape`
    """
    new_shape = []
    for s in shape:
      if isinstance(s, Index):
        new_shape.append(s.dim)
      else:
        new_shape.append(s)

    if np.array_equal(new_shape, self.shape):
      result = self.__new__(type(self))
      result.__init__(
          data=self.data,
          charges=self._charges,
          flows=self._flows,
          order=self._order,
          check_consistency=False)
      return result

    # a few simple checks
    if np.prod(new_shape) != np.prod(self.shape):
      raise ValueError("A tensor with {} elements cannot be "
                       "reshaped into a tensor with {} elements".format(
                           np.prod(self.shape), np.prod(new_shape)))
    flat_dims = np.asarray(
        [self._charges[n].dim for o in self._order for n in o])

    if len(new_shape) > len(self._charges):
      raise ValueError(f"The shape {tuple(new_shape)} is incompatible with the "
                       f"elementary shape {tuple(flat_dims)} of the tensor.")


    if np.any(new_shape == 0) or np.any(flat_dims == 0):
      raise ValueError("reshaping empty arrays is ambiguous, and is currently "
                       "not supported.")

    partitions = [0]
    for n, ns in enumerate(new_shape):
      tmp = np.nonzero(np.cumprod(flat_dims) == ns)[0]
      if len(tmp) == 0:
        raise ValueError(
            "The shape {} is incompatible with the "
            "elementary shape {} of the tensor.".format(
                tuple(new_shape),
                tuple([self._charges[n].dim for o in self._order for n in o])))

      partitions.append(tmp[0] + 1)
      flat_dims = flat_dims[partitions[-1]:]
    for d in flat_dims:
      if d != 1: #Note (martin): I suspect this will never be raised
        raise ValueError(
            "The shape {} is incompatible with the "
            "elementary shape {} of the tensor.".format(
                tuple(new_shape),
                tuple([self._charges[n].dim for o in self._order for n in o])))
      partitions[-1] += 1

    partitions = np.cumsum(partitions)

    flat_order = self.flat_order
    new_order = []
    for n in range(1, len(partitions)):
      new_order.append(list(flat_order[partitions[n - 1]:partitions[n]]))
    result = self.__new__(type(self))
    result.__init__(
        data=self.data,
        charges=self._charges,
        flows=self._flows,
        order=new_order,
        check_consistency=False)
    return result

  def contiguous(self) -> "ChargeArray":
    """
    Transpose the tensor data such that the linear order 
    of the elements in `ChargeArray.data` corresponds to the 
    current order of tensor indices. 
    Consider a tensor with current order given by `_order=[[1,2],[3],[0]]`,
    i.e. `data` was initialized according to order [0,1,2,3], and the tensor
    has since been reshaped and transposed. The linear order of `data` does not
    match the desired order [1,2,3,0] of the tensor. `contiguous` fixes this
    by permuting `data` into this order, transposing `_charges` and `_flows`,
    and changing `_order` to `[[0,1],[2],[3]]`.
    """

    flat_charges = self._charges
    flat_shape = [c.dim for c in flat_charges]
    flat_order = self.flat_order
    tmp = np.append(0, np.cumsum([len(o) for o in self._order]))
    order = [list(np.arange(tmp[n], tmp[n + 1])) for n in range(len(tmp) - 1)]
    data = np.array(
        np.ascontiguousarray(
            np.transpose(np.reshape(self.data, flat_shape), flat_order)).flat)
    result = self.__new__(type(self))
    result.__init__(
        data,
        charges=[self._charges[o] for o in flat_order],
        flows=[self._flows[o] for o in flat_order],
        order=order,
        check_consistency=False)
    return result

  def transpose(self,
                order: Sequence[int] = np.asarray([1, 0]),
                shuffle: Optional[bool] = False) -> "ChargeArray":
    """
    Transpose the tensor into the new order `order`. If `shuffle=False`
    no data-reshuffling is done.
    Args:
      order: The new order of indices.
      shuffle: If `True`, reshuffle data.
    Returns:
      ChargeArray: The transposed tensor.
    """
    if len(order) != self.ndim:
      raise ValueError(
          "`len(order)={}` is different form `self.ndim={}`".format(
              len(order), self.ndim))

    order = [self._order[o] for o in order]
    tensor = self.__new__(type(self))
    tensor.__init__(
        data=self.data,
        charges=self._charges,
        flows=self._flows,
        order=order,
        check_consistency=False)
    if shuffle:
      return tensor.contiguous()
    return tensor

  def conj(self) -> "ChargeArray":
    """
    Complex conjugate operation.
    Returns:
      ChargeArray: The conjugated tensor
    """
    return ChargeArray(
        data=np.conj(self.data),
        charges=self._charges,
        flows=np.logical_not(self._flows),
        order=self._order,
        check_consistency=False)

  @property
  def H(self):
    if self.ndim != 2:
      raise ValueError("hermitian conjugate only works for 2d arrays")
    order = [self._order[o] for o in [1, 0]]
    tensor = self.__new__(type(self))
    tensor.__init__(
        data=np.conj(self.data),
        charges=self._charges,
        flows=self._flows,
        order=order,
        check_consistency=False)

    return tensor

  @property
  def T(self) -> "ChargeArray":
    return self.transpose()
  def __pow__(self, number):
    raise NotImplementedError("__pow__ not implemented for ChargeArray")
  def __sub__(self, other: "BlockSparseTensor") -> "ChargeArray":
    raise NotImplementedError("__sub__ not implemented for ChargeArray")

  def __add__(self, other: "ChargeArray") -> "ChargeArray":
    raise NotImplementedError("__add__ not implemented for ChargeArray")

  def __neg__(self) -> "ChargeArray":
    raise NotImplementedError("__neg__ not implemented for ChargeArray")

  def __mul__(self, number: np.number) -> "ChargeArray":
    raise NotImplementedError("__mul__ not implemented for ChargeArray")

  def __rmul__(self, number: np.number) -> "ChargeArray":
    raise NotImplementedError("__rmul__ not implemented for ChargeArray")

  def __truediv__(self, number: np.number) -> "ChargeArray":
    raise NotImplementedError("__truediv__ not implemented for ChargeArray")

  def __repr__(self):
    if len(self._charges) > 0:
      charge_types = self._charges[0].names
    else:
      charge_types = 'no charge types (scalar)'

    output = self.__class__.__name__ +'\n   shape: ' + repr(
        self.shape
    ) + '\n   charge types: ' + charge_types + '\n   dtype: ' + repr(
        self.dtype.name) + '\n   flat flows: ' + repr(
            self.flat_flows) + '\n   order: ' + repr(self._order)

    return output

  def item(self):
    if len(self.shape) > 0:
      if reduce(mul, self.shape) != 1:
        raise ValueError("can only convert an array of "
                         "size 1 to a Python scalar")
    if len(self.data) == 1:
      return self.data[0]
    return self.dtype.type(0.0)




def compare_shapes(tensor1: ChargeArray, tensor2: ChargeArray) -> bool:
  """
  Compare the shapes of `tensor1` and `tensor2`. Return `True` if the shapes
  are identical.
  Args: 
    tensor1, tensor2: Two tensors.
  Returns:
    bool: The result of comparing the shapes.
  """
  if tensor1.shape != tensor2.shape:
    return False
  if len(tensor1._charges) != len(tensor2._charges):
    return False
  if not all([
      charge_equal(c1, c2) for c1, c2 in zip(tensor1._charges, tensor2._charges)
  ]):
    return False
  if not all([f1 == f2 for f1, f2 in zip(tensor1._flows, tensor2._flows)]):
    return False
  return True


class BlockSparseTensor(ChargeArray):
  """
  A block-sparse tensor class. This class stores non-zero
  elements of a symmetric tensor using an element wise
  encoding.
  The tensor data is stored in a flat np.ndarray `data`.
  Attributes:
    * _data: An np.ndarray containing the data of the tensor.
    * _charges: A list of `BaseCharge` objects, one for each 
        elementary leg of the tensor.
    * _flows: A list of bool, denoting the flow direction of
        each elementary leg.
    * _order: A list of list of int: Used to implement `reshape` and 
        `transpose` operations. Both operations act entirely
        on meta-data of the tensor. `_order` determines which elemetary 
        legs of the tensor are combined, and where they go. 
        E.g. a tensor of rank 4 is initialized with 
        `_order=[[0],[1],[2],[3]]`. Fusing legs 1 and 2
        results in `_order=[[0],[1,2],[3]]`, transposing with 
        `(1,2,0)` results in `_order=[[1,2],[3],[0]]`.
        No data is shuffled during these operations.
  """

  def __init__(self,
               data: np.ndarray,
               charges: List[BaseCharge],
               flows: Union[np.ndarray, List[bool]],
               order: Optional[List[Union[List, np.ndarray]]] = None,
               check_consistency: Optional[bool] = False) -> None:
    """
    Args: 
      data: An np.ndarray containing the actual data. 
      charges: A list of `BaseCharge` objects.
      flows: The flows of the tensor indices, `False` for inflowing, `True`
        for outflowing.
      order: An optional order argument, determining the shape and order of the
        tensor.
      check_consistency: If `True`, check if `len(data)` is consistent with 
        number of non-zero elements given by the charges. This usually causes
        significant overhead, so use only for debugging.
    """
    super().__init__(
        data=data,
        charges=charges,
        flows=flows,
        order=order,
        check_consistency=check_consistency)

    if check_consistency and (len(self._charges) > 0):
      num_non_zero_elements = compute_num_nonzero(self._charges, self._flows)
      if num_non_zero_elements != len(data.flat):
        raise ValueError("number of tensor elements {} defined "
                         "by `charges` is different from"
                         " len(data)={}".format(num_non_zero_elements,
                                                len(data.flat)))

  def copy(self) -> "BlockSparseTensor":
    """
    Return a copy of the tensor.
    """
    return BlockSparseTensor(self.data.copy(),
                             [c.copy() for c in self._charges],
                             self._flows.copy(), copy.deepcopy(self._order),
                             False)

  @classmethod
  def fromdense(cls, indices: List[Index],
                array: np.ndarray) -> "BlockSparseTensor":
    """
    Initialize a BlockSparseTensor from a dense array.
    Args:
      indices: A list of `Index` objects.
      array: A numpy array.
    Returns:
      BlockSparseTensors: A Tensor initialized from the elements
        of `array` at the positions where `indices` fuse to 
        the identity charge.
    """
    shape = [i.dim for i in indices]
    if not np.array_equal(shape, array.shape):
      raise ValueError(
          f"Cannot initialize an BlockSparseTensor of shape {shape}"
          f" from an array of shape {array.shape}")
    tmp = np.append(0, np.cumsum([len(i.flat_charges) for i in indices]))
    order = [list(np.arange(tmp[n], tmp[n + 1])) for n in range(len(tmp) - 1)]

    charges = []
    flows = []
    for i in indices:
      charges.extend(i.flat_charges)
      flows.extend(i.flat_flows)

    _, locs = reduce_charges(
        charges=charges,
        flows=flows,
        target_charges=charges[0].identity_charges(dim=1).unique_charges,
        return_locations=True)

    ar = np.ravel(array)
    data = ar[locs]
    return cls(
        data=data,
        charges=charges,
        flows=flows,
        order=order,
        check_consistency=False)

  def todense(self) -> np.ndarray:
    """
    Map the sparse tensor to dense storage.
    
    """
    if self.ndim == 0:
      return self.data
    out = np.asarray(np.zeros(self.shape, dtype=self.dtype).flat)

    out[np.nonzero(
        fuse_charges(self._charges, self._flows) ==
        self._charges[0].identity_charges(dim=1))[0]] = self.data
    result = np.reshape(out, [c.dim for c in self._charges])
    flat_order = flatten(self._order)
    return result.transpose(flat_order).reshape(self.shape)

  @classmethod
  def randn(cls,
            indices: Sequence[Index],
            dtype: Optional[Type[np.number]] = None) -> "BlockSparseTensor":
    """
    Initialize a random symmetric tensor from a random normal distribution
    with mean 0 and variance 1.
    Args:
      indices: List of `Index` objects, one for each leg. 
      dtype: An optional numpy dtype. The dtype of the tensor
    Returns:
      BlockSparseTensor
    """
    data, charges, flows, order = _data_initializer(_randn,
                                                    compute_num_nonzero,
                                                    indices, dtype=dtype)
    return cls(
        data=data,
        charges=charges,
        flows=flows,
        order=order,
        check_consistency=False)

  @classmethod
  def random(cls,
             indices: Sequence[Index],
             boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
             dtype: Optional[Type[np.number]] = None) -> "BlockSparseTensor":
    """
    Initialize a random symmetric tensor from random uniform distribution.
    Args:
      indices: List of `Index` objects, one for each leg. 
      boundaries: Tuple of interval boundaries for the random uniform 
        distribution.
      dtype: An optional numpy dtype. The dtype of the tensor
    Returns:
      BlockSparseTensor
    """
    data, charges, flows, order = _data_initializer(_random,
                                                    compute_num_nonzero,
                                                    indices, dtype=dtype,
                                                    boundaries=boundaries)
    return cls(
        data=data,
        charges=charges,
        flows=flows,
        order=order,
        check_consistency=False)

  @classmethod
  def ones(cls,
           indices: Sequence[Index],
           dtype: Optional[Type[np.number]] = None) -> "BlockSparseTensor":
    """
    Initialize a symmetric tensor with ones.
    Args:
      indices: List of `Index` objects, one for each leg. 
      dtype: An optional numpy dtype. The dtype of the tensor
    Returns:
      BlockSparseTensor
    """
    data, charges, flows, order = _data_initializer(np.ones,
                                                    compute_num_nonzero,
                                                    indices, dtype=dtype)

    return cls(
        data=data,
        charges=charges,
        flows=flows,
        order=order,
        check_consistency=False)

  @classmethod
  def zeros(cls,
            indices: Sequence[Index],
            dtype: Optional[Type[np.number]] = None) -> "BlockSparseTensor":
    """
    Initialize a symmetric tensor with zeros.
    Args:
      indices: List of `Index` objects, one for each leg. 
      dtype: An optional numpy dtype. The dtype of the tensor
    Returns:
      BlockSparseTensor
    """
    data, charges, flows, order = _data_initializer(np.zeros,
                                                    compute_num_nonzero,
                                                    indices, dtype=dtype)

    return cls(
        data=data,
        charges=charges,
        flows=flows,
        order=order,
        check_consistency=False)

  def _sub_add_protection(self, other):
    if not isinstance(other, type(self)):
      raise TypeError(
          "Can only add or subtract BlockSparseTensor from BlockSparseTensor. "
          "Found type {}".format(type(other)))

    if self.shape != other.shape:
      raise ValueError(
          "cannot add or subtract tensors with shapes {} and {}".format(
              self.shape, other.shape))
    if len(self._charges) != len(other._charges):
      raise ValueError(
          "cannot add or subtract tensors with different charge "
          "lengths {} and {}"
          .format(len(self._charges), len(other._charges)))
    if not np.all([
        self.sparse_shape[n] == other.sparse_shape[n]
        for n in range(len(self.sparse_shape))
    ]):
      raise ValueError(
          "cannot add or subtract tensors non-matching sparse shapes")

  def _align_storage_layout(self, other):
    """
    Align storage layouts of self and other.
    """
    self_is_ordered = np.array_equal(self.flat_order,
                                     np.arange(len(self.flat_order)))
    other_is_ordered = np.array_equal(other.flat_order,
                                      np.arange(len(other.flat_order)))
    both_unordered = (not self_is_ordered) and (not other_is_ordered)
    if self_is_ordered and (not other_is_ordered):
      #bring other into the same storage layout as other
      perm = np.empty(len(other.flat_order), dtype=np.int32)
      perm[self.flat_order] = other.flat_order
      other.contiguous(perm, inplace=True)
    elif ((not self_is_ordered) and other_is_ordered) or both_unordered:
      #bring self into the same storage layout as other
      perm = np.empty(len(self.flat_order), dtype=np.int32)
      perm[other.flat_order] = self.flat_order
      self.contiguous(perm, inplace=True)

  def __sub__(self, other: "BlockSparseTensor") -> "BlockSparseTensor":
    self._sub_add_protection(other)  #perform checks
    self._align_storage_layout(other)  #align storage layout of self and other
    return BlockSparseTensor(
        data=self.data - other.data,
        charges=self._charges,
        flows=self._flows,
        order=self._order,
        check_consistency=False)

  def __add__(self, other: "BlockSparseTensor") -> "BlockSparseTensor":
    self._sub_add_protection(other)  #perform checks
    self._align_storage_layout(other)  #align storage layout of self and other
    #now addition is save
    return BlockSparseTensor(
        data=self.data + other.data,
        charges=self._charges,
        flows=self._flows,
        order=self._order,
        check_consistency=False)

  def __neg__(self) -> "BlockSparseTensor":
    return (-1) * self

  def __mul__(self, number: np.number) -> "BlockSparseTensor":
    if not np.isscalar(number):
      raise TypeError(
          "Can only multiply BlockSparseTensor by a number. Found type {}"
          .format(type(number)))
    return BlockSparseTensor(
        data=self.data * number,
        charges=self._charges,
        flows=self._flows,
        order=self._order,
        check_consistency=False)

  def __rmul__(self, number: np.number) -> "BlockSparseTensor":
    if not np.isscalar(number):
      raise TypeError(
          "Can only right-multiply BlockSparseTensor by a number. Found type {}"
          .format(type(number)))
    return BlockSparseTensor(
        data=self.data * number,
        charges=self._charges,
        flows=self._flows,
        order=self._order,
        check_consistency=False)

  def __truediv__(self, number: np.number) -> "BlockSparseTensor":
    if not np.isscalar(number):
      raise TypeError(
          "Can only divide BlockSparseTensor by a number. Found type {}".format(
              type(number)))

    return BlockSparseTensor(
        data=self.data / number,
        charges=self._charges,
        flows=self._flows,
        order=self._order,
        check_consistency=False)
  
  def __pow__(self, number: np.number) -> "BlockSparseTensor":
    if not np.isscalar(number):
      raise TypeError(
          "Can only exponentiate BlockSparseTensor by a number. Found type {}"
          .format(type(number)))
    return BlockSparseTensor(
        data=self.data ** number,
        charges=self._charges,
        flows=self._flows,
        order=self._order,
        check_consistency=False)


  # pylint: disable=arguments-differ
  def contiguous(self,
                 permutation: Optional[Union[Tuple, List, np.ndarray]] = None,
                 inplace: Optional[bool] = False) -> Any:
    """
    Transpose the tensor data in place such that the linear order 
    of the elements in `BlockSparseTensor.data` corresponds to the 
    current order of tensor indices. 
    Consider a tensor with current order given by `_order=[[1,2],[3],[0]]`,
    i.e. `data` was initialized according to order [0,1,2,3], and the tensor
    has since been reshaped and transposed. The linear oder of `data` does not
    match the desired order [1,2,3,0] of the tensor. `contiguous` fixes this
    by permuting `data` into this order, transposing `_charges` and `_flows`,
    and changing `_order` to `[[0,1],[2],[3]]`.
    Args:
      permutation: An optional alternative order to be used to transposed the 
        tensor. If `None` defaults to `BlockSparseTensor.permutation`.
    """
    flat_charges = self._charges
    flat_flows = self._flows
    if permutation is None:
      permutation = self.flat_order

    if np.array_equal(permutation, np.arange(len(permutation))):
      return self
    tr_partition = _find_best_partition(
        [flat_charges[n].dim for n in permutation])
    tr_sparse_blocks, tr_charges, _ = _find_transposed_diagonal_sparse_blocks(
        flat_charges, flat_flows, tr_partition, permutation)

    sparse_blocks, charges, _ = _find_diagonal_sparse_blocks(
        [flat_charges[n] for n in permutation],
        [flat_flows[n] for n in permutation], tr_partition)
    data = np.empty(len(self.data), dtype=self.dtype)
    for n, sparse_block in enumerate(sparse_blocks):
      ind = np.nonzero(tr_charges == charges[n])[0][0]
      perm = tr_sparse_blocks[ind]
      data[sparse_block] = self.data[perm]

    _, inds = np.unique(permutation, return_index=True)
    new_flat_order = inds[self.flat_order]
    tmp = np.append(0, np.cumsum([len(o) for o in self._order]))
    order = [
        list(new_flat_order[tmp[n]:tmp[n + 1]]) for n in range(len(tmp) - 1)
    ]
    charges = [self._charges[o] for o in permutation]
    flows = np.asarray([self._flows[o] for o in permutation])
    if not inplace:
      return BlockSparseTensor(
          data,
          charges=charges,
          flows=flows,
          order=order,
          check_consistency=False)
    self.data = data
    self._order = order
    self._charges = charges
    self._flows = flows
    return self

  def __matmul__(self, other):

    if (self.ndim > 2) or (other.ndim > 2):
      raise ValueError("__matmul__ is only implemented for vectors or matrices."
                       " Found ndims = {} and {}".format(self.ndim, other.ndim))
    return tensordot(self, other, ([self.ndim - 1], [0]))

  def conj(self) -> "BlockSparseTensor":
    """
    Complex conjugate operation.
    Returns:
      ChargeArray: The conjugated tensor
    """
    return BlockSparseTensor(
        data=np.conj(self.data),
        charges=self._charges,
        flows=list(np.logical_not(self._flows)),
        order=self._order,
        check_consistency=False)

  @property
  def T(self) -> "BlockSparseTensor":
    return self.transpose()


def outerproduct(tensor1: BlockSparseTensor,
                 tensor2: BlockSparseTensor) -> BlockSparseTensor:
  """
  Compute the outer product of two `BlockSparseTensor`.
  The first `tensor1.ndim` indices of the resulting tensor are the 
  indices of `tensor1`, the last `tensor2.ndim` indices are those
  of `tensor2`.
  Args:
    tensor1: A tensor.
    tensor2: A tensor.
  Returns:
    BlockSparseTensor: The result of taking the outer product.
  """

  final_charges = tensor1._charges + tensor2._charges
  final_flows = list(tensor1._flows) + list(tensor2._flows)
  order2 = [list(np.asarray(s) + len(tensor1._charges)) for s in tensor2._order]

  data = np.zeros(
      compute_num_nonzero(final_charges, final_flows), dtype=tensor1.dtype)
  if ((len(tensor1.data) > 0) and (len(tensor2.data) > 0)) and (len(data) > 0):
    # find the location of the zero block in the output
    final_block_maps, final_block_charges, _ = _find_diagonal_sparse_blocks(
        final_charges, final_flows, len(tensor1._charges))
    index = np.nonzero(
        final_block_charges == final_block_charges.identity_charges(
            dim=1))[0][0]
    data[final_block_maps[index].ravel()] = np.outer(tensor1.data,
                                                     tensor2.data).ravel()

  return BlockSparseTensor(
      data,
      charges=final_charges,
      flows=final_flows,
      order=tensor1._order + order2,
      check_consistency=False)


def tensordot(
    tensor1: BlockSparseTensor,
    tensor2: BlockSparseTensor,
    axes: Optional[Union[Sequence[Sequence[int]], Sequence[int], int]] = 2
) -> BlockSparseTensor:
  """
  Contract two `BlockSparseTensor`s along `axes`.
  Args:
    tensor1: First tensor.
    tensor2: Second tensor.
    axes: The axes to contract.
  Returns:
      BlockSparseTensor: The result of the tensor contraction.
  """
  #process scalar input for `axes`
  if isinstance(axes, (np.integer, int)):
    axes = [
        np.arange(tensor1.ndim - axes, tensor1.ndim, dtype=np.int16),
        np.arange(0, axes, dtype=np.int16)
    ]
  elif isinstance(axes[0], (np.integer, int)):
    if len(axes) > 1:
      raise ValueError("invalid input `axes = {}` to tensordot".format(axes))
    axes = [np.array(axes, dtype=np.int16), np.array(axes, dtype=np.int16)]
  axes1 = axes[0]
  axes2 = axes[1]

  if len(axes1) != len(axes2):
    raise ValueError(
        "`axes1 = {}` and `axes2 = {}` have to be of same length. ".format(
            axes1, axes2))

  if len(axes1) > len(tensor1.shape):
    raise ValueError(
        "`axes1 = {}` is incompatible with `tensor1.shape = {}. ".format(
            axes1, tensor1.shape))

  if len(axes2) > len(tensor2.shape):
    raise ValueError(
        "`axes2 = {}` is incompatible with `tensor2.shape = {}. ".format(
            axes2, tensor2.shape))

  #special case outer product
  if len(axes1) == 0:
    return outerproduct(tensor1, tensor2)

  #more checks
  if max(axes1) >= len(tensor1.shape):
    raise ValueError(
        "rank of `tensor1` is smaller than `max(axes1) = {}.`".format(
            max(axes1)))

  if max(axes2) >= len(tensor2.shape):
    raise ValueError(
        "rank of `tensor2` is smaller than `max(axes2) = {}`".format(
            max(axes1)))

  contr_flows_1 = []
  contr_flows_2 = []
  contr_charges_1 = []
  contr_charges_2 = []
  for a in axes1:
    contr_flows_1.extend(tensor1._flows[tensor1._order[a]])
    contr_charges_1.extend([tensor1._charges[n] for n in tensor1._order[a]])
  for a in axes2:
    contr_flows_2.extend(tensor2._flows[tensor2._order[a]])
    contr_charges_2.extend([tensor2._charges[n] for n in tensor2._order[a]])

  if len(contr_charges_2) != len(contr_charges_1):
    raise ValueError(
        "`axes1 = {}` and `axes2 = {}` have incompatible elementary"
        " shapes {} and {}".format(axes1, axes2,
                                   [e.dim for e in contr_charges_1],
                                   [e.dim for e in contr_charges_2]))
  if not np.all(
      np.asarray(contr_flows_1) == np.logical_not(np.asarray(contr_flows_2))):

    raise ValueError(
        "`axes1 = {}` and `axes2 = {}` have incompatible elementary"
        " flows {} and {}".format(axes1, axes2, contr_flows_1, contr_flows_2))
  charge_check = [
      charge_equal(c1, c2) for c1, c2 in zip(contr_charges_1, contr_charges_2)
  ]
  if not np.all(charge_check):
    inds = np.nonzero(np.logical_not(charge_check))[0]
    raise ValueError(
        "`axes = {}` of tensor1 and `axes = {}` of tensor2 have "
        "incompatible charges {} and {}".format(
            np.array(axes1)[inds],
            np.array(axes2)[inds], [contr_charges_1[i] for i in inds],
            [contr_charges_2[i] for i in inds]))

  #checks finished

  #special case inner product (returns an ndim=0 tensor)
  if (len(axes1) == tensor1.ndim) and (len(axes2) == tensor2.ndim):
    t1 = tensor1.transpose(axes1).contiguous()
    t2 = tensor2.transpose(axes2).contiguous()
    return BlockSparseTensor(
        data=np.dot(t1.data, t2.data),
        charges=[],
        flows=[],
        order=[],
        check_consistency=False)

  #in all other cases we perform a regular tensordot
  free_axes1 = sorted(set(np.arange(tensor1.ndim)) - set(axes1))
  free_axes2 = sorted(set(np.arange(tensor2.ndim)) - set(axes2))

  new_order1 = [tensor1._order[n] for n in free_axes1
               ] + [tensor1._order[n] for n in axes1]
  new_order2 = [tensor2._order[n] for n in axes2
               ] + [tensor2._order[n] for n in free_axes2]

  flat_order_1 = flatten(new_order1)
  flat_order_2 = flatten(new_order2)

  flat_charges_1, flat_flows_1 = tensor1._charges, tensor1._flows
  flat_charges_2, flat_flows_2 = tensor2._charges, tensor2._flows

  left_charges = []
  right_charges = []
  left_flows = []
  right_flows = []
  left_order = []
  right_order = []

  s = 0
  for n in free_axes1:
    left_charges.extend([tensor1._charges[o] for o in tensor1._order[n]])
    left_order.append(list(np.arange(s, s + len(tensor1._order[n]))))
    s += len(tensor1._order[n])
    left_flows.extend([tensor1._flows[o] for o in tensor1._order[n]])

  s = 0
  for n in free_axes2:
    right_charges.extend([tensor2._charges[o] for o in tensor2._order[n]])
    right_order.append(
        list(len(left_charges) + np.arange(s, s + len(tensor2._order[n]))))
    s += len(tensor2._order[n])
    right_flows.extend([tensor2._flows[o] for o in tensor2._order[n]])

  tr_sparse_blocks_1, charges1, shapes_1 = _find_transposed_diagonal_sparse_blocks(#pylint: disable=line-too-long
      flat_charges_1, flat_flows_1, len(left_charges), flat_order_1)

  tr_sparse_blocks_2, charges2, shapes_2 = _find_transposed_diagonal_sparse_blocks(#pylint: disable=line-too-long
      flat_charges_2, flat_flows_2, len(contr_charges_2), flat_order_2)

  common_charges, label_to_common_1, label_to_common_2 = intersect(
      charges1.unique_charges,
      charges2.unique_charges,
      axis=0,
      return_indices=True)

  #Note: `cs` may contain charges that are not present in `common_charges`
  charges = left_charges + right_charges
  flows = left_flows + right_flows

  sparse_blocks, cs, _ = _find_transposed_diagonal_sparse_blocks(
      charges, flows, len(left_charges), list(range(len(charges))))
  num_nonzero_elements = np.int64(np.sum([len(v) for v in sparse_blocks]))

  #Note that empty is not a viable choice here.
  data = np.zeros(
      num_nonzero_elements, dtype=np.result_type(tensor1.dtype, tensor2.dtype))

  label_to_common_final = intersect(
      cs.unique_charges, common_charges, axis=0, return_indices=True)[1]

  for n in range(common_charges.shape[0]):
    n1 = label_to_common_1[n]
    n2 = label_to_common_2[n]
    nf = label_to_common_final[n]
    data[sparse_blocks[nf].ravel()] = np.ravel(
        np.matmul(tensor1.data[tr_sparse_blocks_1[n1].reshape(shapes_1[:, n1])],
                  tensor2.data[tr_sparse_blocks_2[n2].reshape(shapes_2[:,
                                                                       n2])]))
  res = BlockSparseTensor(
      data=data,
      charges=charges,
      flows=flows,
      order=left_order + right_order,
      check_consistency=False)
  return res
