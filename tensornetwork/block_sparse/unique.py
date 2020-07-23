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
from typing import List, Type, Any


def get_dtype(itemsize):
  final_dtype = np.int8
  if itemsize > 1:
    final_dtype = np.int16
  if itemsize > 2:
    final_dtype = np.int32
  if itemsize > 4:
    final_dtype = np.int64
  return final_dtype


def collapse(array: np.ndarray):
  array = np.ascontiguousarray(array)
  newdtype = get_dtype(array.dtype.itemsize * array.shape[1])
  if array.shape[1] < 5:
    if array.shape[1] in (1, 2, 4):
      tmparray = array.view(newdtype)
    elif array.shape[1] == 3:
      tmparray = np.squeeze(
          np.concatenate([
              array,
              np.full((array.shape[0], 1), fill_value=0, dtype=array.dtype)
          ],
                         axis=1).view(newdtype))
    return np.squeeze(tmparray)
  return array


def expand(array: np.ndarray, original_dtype):
  if array.ndim == 1:
    return array[:, None].view(original_dtype)
  return array


def unique(array: np.ndarray,
           return_index: bool = False,
           return_inverse: bool = False,
           return_counts: bool = False,
           label_dtype: Type[np.number] = np.int16):

  collapsed_array = collapse(array)
  if array.shape[1] < 5:
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
    out[0] = expand(out[0], array.dtype)
    if array.shape[1] == 3:
      out[0] = out[0][:, 0:3]
    if return_inverse and not return_index:
      out[1] = out[1].astype(label_dtype)
    elif return_inverse and return_index:
      out[2] = out[2].astype(label_dtype)
    out[0] = np.ascontiguousarray(out[0])
    
  else:
    if _return_index:
      out = expand(res[0], array.dtype)
      if array.shape[1] == 3:
        out = np.ascontiguousarray(out[:, 0:3])
    else:
      out = expand(res, array.dtype)
      if array.shape[1] == 3:
        out = out[:, 0:3]
      out = np.ascontiguousarray(out)

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

  if A.ndim != B.ndim:
    raise ValueError("array ndims must match to intersect")

  if A.ndim == 1:
    if A.dtype in (np.int8, np.int16) and B.dtype in (np.int8, np.int16):
      out = np.intersect1d(
          A, B, assume_unique=assume_unique, return_indices=True)
      if return_indices:
        result = out
      else:
        result = out[0]
    else:
      result = np.intersect1d(
          A, B, assume_unique=assume_unique, return_indices=return_indices)
    return result

  if A.shape[1] > 4:
    return _intersect_ndarray(A, B, axis, assume_unique, return_indices)

  if A.ndim == 2:
    if axis == 0:
      if A.shape[1] != B.shape[1]:
        raise ValueError("array widths must match to intersect")
      collapsed_A = collapse(A)
      collapsed_B = collapse(B)
      if collapsed_A.dtype in (np.int8,
                               np.int16) and collapsed_B.dtype in (np.int8,
                                                                   np.int16):
        C, A_locs, B_locs = np.intersect1d(
            collapsed_A,
            collapsed_B,
            assume_unique=assume_unique,
            return_indices=True)
        C = expand(C, A.dtype)
        if return_indices:
          result = C, A_locs, B_locs
        else:
          result = C
        return result

      C, A_locs, B_locs = np.intersect1d(
          collapsed_A,
          collapsed_B,
          assume_unique=assume_unique,
          return_indices=return_indices)
      C = expand(C, A.dtype)
      if A.shape[1] == 3:
        C = np.ascontiguousarray(C[:, 0:3])
      return C, A_locs, B_locs

    if axis == 1:
      out = intersect(
          A.T,
          B.T,
          axis=0,
          assume_unique=assume_unique,
          return_indices=return_indices)
      if return_indices:
        return np.ascontiguousarray(out[0].T), out[1], out[2]
      return np.ascontiguousarray(out.T)
    raise NotImplementedError(
        "intersection can only be performed on first or second axis")

  raise NotImplementedError("intersect is only implemented for 1d or 2d arrays")


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
      out = intersect(
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

  raise NotImplementedError("intersect is only implemented for 1d or 2d arrays")
