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

import importlib


def get_backend(name, dtype):

  if name == "tensorflow":
    from .tensorflow import tensorflow_backend
    backend_class = tensorflow_backend.TensorFlowBackend
    dtypes = tensorflow_backend.supported_dtypes
  elif name == "numpy":
    from .numpy import numpy_backend
    backend_class = numpy_backend.NumPyBackend
    dtypes = numpy_backend.supported_dtypes
  elif name == "jax":
    from .jax import jax_backend
    backend_class = jax_backend.JaxBackend
    dtypes = jax_backend.supported_dtypes
  elif name == "shell":
    from .shell import shell_backend
    backend_class = shell_backend.ShellBackend
    dtypes = shell_backend.supported_dtypes
  elif name == "pytorch":
    from .pytorch import pytorch_backend
    backend_class = pytorch_backend.PyTorchBackend
    dtypes = pytorch_backend.supported_dtypes
  else:
    raise ValueError("Backend {} does not exist".format(name))

  if dtype and not any([dtype is d for d in dtypes]):
    raise TypeError("Backend {} does not support dtype={} of type {}".format(
        name, dtype, type(dtype)))

  return backend_class(dtype)
