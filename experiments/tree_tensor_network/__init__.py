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
from experiments.tree_tensor_network import ttn_1d_uniform 
from experiments.tree_tensor_network.ttn_1d_uniform import *

def set_backend(backend_name):
  """Sets the backend to use for tree tensor network computations.

  A backend must be set after importing the module.

  Args:
    backend: Possible values are "tensorflow", "jax", and "numpy".
  """
  if backend_name == "tensorflow":
    ttn_1d_uniform.backend = TTNBackendTensorFlow()
  elif backend_name == "jax":
    ttn_1d_uniform.backend = TTNBackendJAX()
  elif backend_name == "numpy":
    ttn_1d_uniform.backend = TTNBackendNumpy()
  else:
    raise ValueError("Unsupported backend: {}".format(backend_name))
