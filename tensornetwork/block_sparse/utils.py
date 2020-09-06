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
from tensornetwork.block_sparse.sizetypes import SIZE_T
from tensornetwork.block_sparse.caching import get_cacher
from typing import List, Union, Any, Type, Tuple


def _randn(size: int, dtype: Type[np.number] = np.float64) -> np.ndarray:
  """
  Initialize a 1d np.ndarray of length `size` of dtype `dtype`
  with random gaussian values.
  Args:
    size: The length of the array.
    dtype: The desired dtype.
  Returns:
    np.ndarray: The data array.
  """
  data = np.random.randn(size).astype(dtype)
  if ((np.dtype(dtype) is np.dtype(np.complex128)) or
      (np.dtype(dtype) is np.dtype(np.complex64))):
    data += 1j * np.random.randn(size).astype(dtype)
  return data


def _random(size: int,
            dtype: Type[np.number] = np.float64,
            boundaries: Tuple = (0, 1)) -> np.ndarray:
  """
  Initialize a 1d np.ndarray of length `size` of dtype `dtype`
  with random uniform values.
  Args:
    size: The length of the array.
    dtype: The desired dtype.
    boundaries: The boundaries of the interval where numbers are
      drawn from.
  Returns:
    np.ndarray: The data array.
  """

  data = np.random.uniform(boundaries[0], boundaries[1], size).astype(dtype)
  if ((np.dtype(dtype) is np.dtype(np.complex128)) or
      (np.dtype(dtype) is np.dtype(np.complex64))):
    data += 1j * np.random.uniform(boundaries[0], boundaries[1],
                                   size).astype(dtype)
  return data


def get_real_dtype(dtype: Type[np.number]) -> Type[np.number]:
  if dtype == np.complex128:
    return np.float64
  if dtype == np.complex64:
    return np.float32
  return dtype


def flatten(list_of_list: List[List]) -> np.ndarray:
  """
  Flatten a list of lists into a single list.
  Args:
    list_of_lists: A list of lists.
  Returns:
    list: The flattened input.
  """
  return np.array([l for sublist in list_of_list for l in sublist])


def fuse_stride_arrays(dims: Union[List[int], np.ndarray],
                       strides: Union[List[int], np.ndarray]) -> np.ndarray:
  """
  Compute linear positions of tensor elements 
  of a tensor with dimensions `dims` according to `strides`.
  Args: 
    dims: An np.ndarray of (original) tensor dimensions.
    strides: An np.ndarray of (possibly permuted) strides.
  Returns:
    np.ndarray: Linear positions of tensor elements according to `strides`.
  """
  return fuse_ndarrays([
      np.arange(0, strides[n] * dims[n], strides[n], dtype=SIZE_T)
      for n in range(len(dims))
  ])

def fuse_ndarrays(arrays: List[Union[List, np.ndarray]]) -> np.ndarray:
  """
  Fuse all `arrays` by simple kronecker addition.
  Arrays are fused from "right to left", 
  Args:
    arrays: A list of arrays to be fused.
  Returns:
    np.ndarray: The result of fusing `arrays`.
  """
  if len(arrays) == 1:
    return np.array(arrays[0])
  fused_arrays = np.asarray(arrays[0])
  for n in range(1, len(arrays)):
    fused_arrays = np.ravel(np.add.outer(fused_arrays, arrays[n]))
  return fused_arrays

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
  return np.reshape(
      np.array(degen1)[:, None] * np.array(degen2)[None, :],
      len(degen1) * len(degen2))

def _get_strides(dims: Union[List[int], np.ndarray]) -> np.ndarray:
  """
  compute strides of `dims`.
  """
  return np.flip(np.append(1, np.cumprod(np.flip(dims[1::]))))


def _find_best_partition(dims: Union[List[int], np.ndarray]) -> int:
  """
  Find the most-levelled partition of `dims`.
  A levelled partitioning is a partitioning such that
  np.prod(dim[:partition]) and np.prod(dim[partition:])
  are as close as possible.
  Args:
    dims: A list or np.ndarray of integers.
  Returns:
    int: The best partitioning.
  """
  if len(dims) == 1:
    raise ValueError(
        'expecting dims with a length of at least 2, got len(dims) =1')
  diffs = [
      np.abs(np.prod(dims[0:n]) - np.prod(dims[n::]))
      for n in range(1, len(dims))
  ]
  min_inds = np.nonzero(diffs == np.min(diffs))[0]
  if len(min_inds) > 1:
    right_dims = [np.prod(dims[min_ind + 1:]) for min_ind in min_inds]
    min_ind = min_inds[np.argmax(right_dims)]
  else:
    min_ind = min_inds[0]
  return min_ind + 1


def get_dtype(itemsize: int) -> Type[np.number]:
  """
  Return the `numpy.dtype` needed to store an 
  element of `itemsize` bytes.
  """
  final_dtype = np.int8
  if itemsize > 1:
    final_dtype = np.int16
  if itemsize > 2:
    final_dtype = np.int32
  if itemsize > 4:
    final_dtype = np.int64
  return final_dtype


def collapse(array: np.ndarray) -> np.ndarray:
  """
  If possible, collapse a 2d numpy array 
  `array` along the rows into a 1d array of larger 
  dtype.
  Args:
    array: np.ndarray
  Returns:
    np.ndarray: The collapsed array.
  """

  if array.ndim <= 1 or array.dtype.itemsize * array.shape[1] > 8:
    return array
  array = np.ascontiguousarray(array)  
  newdtype = get_dtype(array.dtype.itemsize * array.shape[1])
  if array.shape[1] in (1, 2, 4, 8):
    tmparray = array.view(newdtype)
  else:
    if array.shape[1] == 3:
      width = 1
    else:
      width = 8 - array.shape[1]

    tmparray = np.squeeze(
        np.concatenate(
            [array, np.zeros((array.shape[0], width), dtype=array.dtype)],
            axis=1).view(newdtype))
  return np.squeeze(tmparray)


def expand(array: np.ndarray, original_dtype: Type[np.number],
           original_width: int, original_ndim: int) -> np.ndarray:
  """
  Reverse operation to `collapse`. 
  Expand a 1d numpy array `array` into a 2d array
  of dtype `original_dtype` by view-casting.
  Args:
    array: The collapsed array.
    original_dtype: The dtype of the original (uncollapsed) array
    original_width: The width (the length of the second dimension)
      of the original (uncollapsed) array.
    original_ndim: Number of dimensions of the original (uncollapsed)
      array.
  Returns:
    np.ndarray: The expanded array.
  """
  if original_ndim <= 1:
    #nothing to expand
    return np.squeeze(array)
  if array.ndim == 1:
    #the array has been collapsed
    #now we uncollapse it
    result = array[:, None].view(original_dtype)
    if original_width in (3, 5, 6, 7):
      result = np.ascontiguousarray(result[:, :original_width])
    return result
  return array


def unique(array: np.ndarray,
           return_index: bool = False,
           return_inverse: bool = False,
           return_counts: bool = False,
           label_dtype: Type[np.number] = np.int16) -> Any:
  """
  Compute the unique elements of 1d or 2d `array` along the 
  zero axis of the array.
  This function performs performs a similar
  task to `numpy.unique` with `axis=0` argument,
  but is substantially faster for 2d arrays. 
  Note that for the case of 2d arrays, the ordering of the array of unique 
  elements differs from the ordering of `numpy.unique`.
  Args:
    array: An input array of integers.
    return_index: If `True`, also return the indices of `array` 
      that result in the unique array.
    return_inverse: If `True`, also return the indices of the unique array 
      that can be used to reconstruct `array`.
    return_counts: If `True`, also return the number of times 
      each unique item appears in `array`.
  Returns:
    np.ndarray: An array of unique elements.
    np.ndarray (optional): The indices of array that result 
      in the unique array.
    np.ndarray: (optional): The indices of the unique array
      from which `array` can be reconstructed.
    np.ndarray (optional): The number of times each element of the 
      unique array appears in `array`.
  """
  array = np.asarray(array)
  original_width = array.shape[1]  if array.ndim == 2 else 0
  original_ndim = array.ndim
  collapsed_array = collapse(array)
  if collapsed_array.ndim <= 1:
    axis = None
  else:
    axis = 0

  _return_index = (collapsed_array.dtype in (np.int8, np.int16)) or return_index
  res = np.unique(
      collapsed_array,
      return_index=_return_index,
      return_inverse=return_inverse,
      return_counts=return_counts,
      axis=axis)
  if any([return_index, return_inverse, return_counts]):
    out = list(res)
    if _return_index and not return_index:
      del out[1]
    out[0] = expand(out[0], array.dtype, original_width, original_ndim)
    if return_inverse and not return_index:
      out[1] = out[1].astype(label_dtype)
    elif return_inverse and return_index:
      out[2] = out[2].astype(label_dtype)
    out[0] = np.ascontiguousarray(out[0])
  else:
    if _return_index:
      out = expand(res[0], array.dtype, original_width, original_ndim)
    else:
      out = expand(res, array.dtype, original_width, original_ndim)

  return out


def intersect(A: np.ndarray,
              B: np.ndarray,
              axis=0,
              assume_unique=False,
              return_indices=False) -> Any:
  """
  Extends numpy's intersect1d to find the row or column-wise intersection of
  two 2d arrays. Takes identical input to numpy intersect1d.
  Args:
    A, B (np.ndarray): arrays of matching widths and datatypes
  Returns:
    ndarray: sorted 1D array of common rows/cols between the input arrays
    ndarray: the indices of the first occurrences of the common values in A.
      Only provided if return_indices is True.
    ndarray: the indices of the first occurrences of the common values in B.
      Only provided if return_indices is True.
  """
  if A.dtype != B.dtype:
    raise ValueError(f"array dtypes must macht to intersect, "
                     f"found A.dtype = {A.dtype}, B.dtype = {B.dtype}")
  if axis not in (0, 1):
    raise NotImplementedError(
        "intersection can only be performed on first or second axis")

  if A.ndim != B.ndim:
    raise ValueError("array ndims must match to intersect")

  if axis == 1:
    if A.shape[0] != B.shape[0]:
      raise ValueError("array heights must match to intersect on second axis")

    out = intersect(
        A.T,
        B.T,
        axis=0,
        assume_unique=assume_unique,
        return_indices=return_indices)
    if return_indices:
      return np.ascontiguousarray(out[0].T), out[1], out[2]
    return np.ascontiguousarray(out.T)

  if A.ndim > 1 and A.shape[1] != B.shape[1]:
    raise ValueError("array widths must match to intersect on first axis")
  original_width = A.shape[1] if A.ndim == 2 else 0
  original_ndim = A.ndim
  collapsed_A = collapse(A)
  collapsed_B = collapse(B)

  if collapsed_A.ndim > 1:
    # arrays were not callapsable, fall back to slower implementation
    return _intersect_ndarray(collapsed_A, collapsed_B, axis, assume_unique,
                              return_indices)
  if collapsed_A.dtype in (np.int8,
                           np.int16) and collapsed_B.dtype in (np.int8,
                                                               np.int16):
    #special case of dtype = np.int8 or np.int16
    #original charges were unpadded in this case
    C, A_locs, B_locs = np.intersect1d(
        collapsed_A,
        collapsed_B,
        assume_unique=assume_unique,
        return_indices=True)
    C = expand(C, A.dtype, original_width, original_ndim)
    if return_indices:
      result = C, A_locs, B_locs
    else:
      result = C
  else:
    result = np.intersect1d(
        collapsed_A,
        collapsed_B,
        assume_unique=assume_unique,
        return_indices=return_indices)
    if return_indices:
      result = list(result)
      result[0] = expand(result[0], A.dtype, original_width, original_ndim)
    else:
      result = expand(result, A.dtype, original_width, original_ndim)
  return result



def _intersect_ndarray(A: np.ndarray,
                       B: np.ndarray,
                       axis=0,
                       assume_unique=False,
                       return_indices=False) -> Any:
  """
  Extends numpy's intersect1d to find the row or column-wise intersection of
  two 2d arrays. Takes identical input to numpy intersect1d.
  Args:
    A, B (np.ndarray): arrays of matching widths and datatypes
  Returns:
    ndarray: sorted 1D array of common rows/cols between the input arrays
    ndarray: the indices of the first occurrences of the common values in A.
      Only provided if return_indices is True.
    ndarray: the indices of the first occurrences of the common values in B.
      Only provided if return_indices is True.
  """
  # pylint: disable=line-too-long
  # see
  # https://stackoverflow.com/questions/8317022/ get-intersecting-rows-across-two-2d-numpy-arrays
  #pylint: disable=no-else-return
  A = np.ascontiguousarray(A)
  B = np.ascontiguousarray(B)
  if A.ndim != B.ndim:
    raise ValueError("array ndims must match to intersect")
  if A.ndim == 1:
    return np.intersect1d(
        A, B, assume_unique=assume_unique, return_indices=return_indices)

  elif A.ndim == 2:
    if axis == 0:
      ncols = A.shape[1]
      if A.shape[1] != B.shape[1]:
        raise ValueError("array widths must match to intersect")

      dtype = {
          'names': ['f{}'.format(i) for i in range(ncols)],
          'formats': ncols * [A.dtype]
      }
      if return_indices:
        C, A_locs, B_locs = np.intersect1d(
            A.view(dtype),
            B.view(dtype),
            assume_unique=assume_unique,
            return_indices=return_indices)
        return C.view(A.dtype).reshape(-1, ncols), A_locs, B_locs
      C = np.intersect1d(
          A.view(dtype), B.view(dtype), assume_unique=assume_unique)
      return C.view(A.dtype).reshape(-1, ncols)

    elif axis == 1:
      out = _intersect_ndarray(
          A.T.copy(),
          B.T.copy(),
          axis=0,
          assume_unique=assume_unique,
          return_indices=return_indices)
      if return_indices:
        return np.ascontiguousarray(out[0].T), out[1], out[2]
      return np.ascontiguousarray(out.T)

    raise NotImplementedError(
        "intersection can only be performed on first or second axis")

  raise NotImplementedError("_intersect_ndarray is only implemented for 1d or 2d arrays")
