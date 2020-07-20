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
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.charge import (fuse_charges, fuse_degeneracies,
                                               BaseCharge, fuse_ndarray_charges,
                                               intersect, charge_equal,
                                               fuse_ndarrays)
from typing import List, Union, Any, Tuple, Optional, Sequence
_INSTANTIATED_CACHERS = []
class Cacher:
  def __init__(self):
    self.cache = {}
    self.do_caching = False
    
  def set_status(self, value):
    self.do_caching = value
    
  def clear_cache(self):
    self.cache = {}
  
def get_cacher():
  """
  Return a `Cacher` object which can be used to perform 
  caching of block-data for block-sparse tensor contractions.
  """
  if len(_INSTANTIATED_CACHERS) == 0:
    _INSTANTIATED_CACHERS.append(Cacher())
  return _INSTANTIATED_CACHERS[0]
    
def enable_caching():
    """
    Enable caching of block-data for block-sparse contraction.
    If enabled, all data that is needed to perform binary tensor contractions 
    will be cached in a dictionary for later reuse. 
    Enabling caching can significantly speed tensor contractions,
    but can lead to substantially larger memory footprints.
    In particular if the code uses tensor decompositions like QR, SVD
    eig, eigh or any similar method, enabling caching can cause catastrophic memory 
    clutter, so be careful when turning it on.

    The user can at any point clear the cache by calling 
    `tn.block_sparse.clear_cache()`.
    """
  _INSTANTIATED_CACHERS[0].set_status(True)
  
def disable_caching():
  """
  Disable caching of block-data for block-sparse tensor contractions. 
  Note that the cache WILL NOT BE CLEARED. 
  Clearing the cache can be achieved by calling
  `tn.block_sparse.clear_cache()`.
  """
  _INSTANTIATED_CACHERS[0].set_status(False)

def clear_cache():
  """
  Clear the cache that stores block-data for block-sparse tensor contractions.
  """
  _INSTANTIATED_CACHERS[0].clear_cache()


  
