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
default_backend = "jax"
default_dtype = None
#for backwards compatibility default dtypes have to be `None`
#changing this will cause tests to fail due backend.convert_to_tensor(tensor)
#raising TypeErrors when incoming `tensor` has a dtype different from
#backend.dtype.

default_dtypes = {
    'tensorflow': None,
    'numpy': None,
    'pytorch': None,
    'jax': None,
    'shell': None
}
supported_numpy_dtypes = [
    np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64,
    np.complex64, np.complex128, np.bool
]

supported_jax_dtypes = [
    np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64,
    np.complex64, np.complex128, np.bool
]
#np.ndarray.dtype doesn't return a `type`, but a `dtype` object
#backend_factory compares object identities, and we want to catch both
#the case of `dtype` being a `type` (like np.float64 given by a user) AND
#the case of `dtype` being obtained from `np.ndarray.dtype`
numpy_dtypes = supported_numpy_dtypes + [
    np.dtype(d) for d in supported_numpy_dtypes
]

#shell supports everything
supported_shell_dtypes = supported_numpy_dtypes

supported_dtypes = {
    'numpy': numpy_dtypes + [None],
    'jax': numpy_dtypes + [None],
    'shell': supported_shell_dtypes + [None]
}

try:
  import torch
  supported_pytorch_dtypes = [
      torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
      torch.float32, torch.float64, torch.bool
  ]
except ImportError:
  supported_pytorch_dtypes = []

try:
  import tensorflow as tf
  supported_tensorflow_dtypes = [
      tf.int8, tf.int16, tf.int32, tf.int64, tf.float32, tf.float16, tf.float64,
      tf.complex64, tf.complex128, tf.bool
  ]
except ImportError:
  supported_tensorflow_dtypes = []

#shell supports everything
supported_shell_dtypes = supported_tensorflow_dtypes + \
    supported_numpy_dtypes + supported_pytorch_dtypes

supported_dtypes = {
    'tensorflow': supported_tensorflow_dtypes + [None],
    'numpy': numpy_dtypes + [None],
    'pytorch': supported_pytorch_dtypes + [None],
    'jax': numpy_dtypes + [None],
    'shell': supported_shell_dtypes + [None]
}
