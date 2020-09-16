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

from typing import Union, Text
from tensornetwork.backends.tensorflow import tensorflow_backend
from tensornetwork.backends.numpy import numpy_backend
from tensornetwork.backends.jax import jax_backend
from tensornetwork.backends.pytorch import pytorch_backend
from tensornetwork.backends.symmetric import symmetric_backend
from tensornetwork.backends import abstract_backend
_BACKENDS = {
    "tensorflow": tensorflow_backend.TensorFlowBackend,
    "numpy": numpy_backend.NumPyBackend,
    "jax": jax_backend.JaxBackend,
    "pytorch": pytorch_backend.PyTorchBackend,
    "symmetric": symmetric_backend.SymmetricBackend
}

#we instantiate each backend only once and store it here
_INSTANTIATED_BACKENDS = dict()


def get_backend(
    backend: Union[Text, abstract_backend.AbstractBackend]
) -> abstract_backend.AbstractBackend:
  if isinstance(backend, abstract_backend.AbstractBackend):
    return backend
  if backend not in _BACKENDS:
    raise ValueError("Backend '{}' does not exist".format(backend))

  if backend in _INSTANTIATED_BACKENDS:
    return _INSTANTIATED_BACKENDS[backend]

  _INSTANTIATED_BACKENDS[backend] = _BACKENDS[backend]()
  return _INSTANTIATED_BACKENDS[backend]
