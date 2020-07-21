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
from typing import List, Type


def get_dtype(itemsize):
  final_dtype = np.int8
  if itemsize > 1:
    final_dtype = np.int16
  if itemsize > 2:
    final_dtype = np.int32
  if itemsize > 4:
    final_dtype = np.int64
  return final_dtype


def unique(array: np.ndarray,
           return_index: bool = False,
           return_inverse: bool = False,
           return_counts: bool = False,
           label_dtype: Type[np.number] = np.int16):
  newdtype = get_dtype(array.dtype.itemsize * array.shape[1])
  if array.shape[1] < 5:
    if array.shape[1] in (1, 2, 4):
      tmparray = array.view(newdtype)
      dim = array.shape[1]
    elif array.shape[1] == 3:
      tmparray = np.squeeze(
          np.concatenate([
              array,
              np.full((array.shape[0], 1), fill_value=0, dtype=array.dtype)
          ],
                         axis=1).view(newdtype))
      dim = 4
    
    out = np.unique(
        tmparray,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts)
    
    if any([return_index, return_inverse, return_counts]):
      out = list(out)
      out[0] = np.reshape(out[0].view(array.dtype), (out[0].shape[0], dim))
      if array.shape[1] == 3:
        out[0] = out[0][:, 0:3]
      if return_inverse and not return_index:
        out[1] = out[1].astype(label_dtype)
      elif return_inverse and return_index:
        out[2] = out[2].astype(label_dtype)
    else:
      out = np.reshape(out.view(array.dtype), (out.shape[0], dim))
      if array.shape[1] == 3:
        out = out[:, 0:3]
  else:
    out = np.unique(
        tmparray,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts, axis=0)
    if any([return_index, return_inverse, return_counts]):
      out = list(out)
      if return_inverse and not return_index:
        out[1] = out[1].astype(label_dtype)
      elif return_inverse and return_index:
        out[2] = out[2].astype(label_dtype)
      
  return out
