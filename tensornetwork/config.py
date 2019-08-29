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
import tensorflow as tf
import numpy as np
import torch
default_backend = "tensorflow"
default_dtypes = {
    'tensorflow': tf.float64,
    'numpy': np.float64,
    'pytorch': torch.float64,
    'jax': np.float64,
    'shell': np.float64
}

supported_dtypes = {
    'tensorflow': (tf.int8, tf.int16, tf.int32, tf.int64, tf.float32,
                   tf.float16, tf.float64, tf.complex64, tf.complex128,
                   tf.bool),
    'numpy': (np.int8, np.int16, np.int32, np.int64, np.float16, np.float32,
              np.float64, np.complex64, np.complex128, np.bool),
    'pytorch': (torch.int8, torch.int16, torch.int32, torch.int64,
                torch.float16, torch.float32, torch.float64, torch.complex64,
                torch.complex128, torch.bool),
    'jax': (np.int8, np.int16, np.int32, np.int64, np.float16, np.float32,
            np.float64, np.complex64, np.complex128, np.bool),
}
#shell supports everything
supported_dtypes['shell'] =\
    supported_dtypes['numpy'] + \
    supported_dtypes['pytorch'] + \
    supported_dtypes['tensorflow']
