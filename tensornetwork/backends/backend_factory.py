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

jax_config_args = dict(
    dtype=None,  # default: float64 
    precision=None  # default: jax.lax.Precision.DEFAULT
)
numpy_config_args = dict()
pytorch_config_args = dict()
tensorflow_config_args = dict()
symmetric_config_args = dict()

CONFIG_ARGS = {
    'numpy': numpy_config_args,
    'jax': jax_config_args,
    'tensorflow': tensorflow_config_args,
    'pytorch': pytorch_config_args,
    'symmetric': symmetric_config_args
}


def configure_backend(backend: Union[Text, abstract_backend.AbstractBackend],
                      **kwargs):
  """
  Globally configure the behaviour of different backends.
  All backends in _INSTANTIATED_BACKENDS are reconfigured, 
  and the default config parameters will be changed to `**kwargs`.
  """
  if isinstance(backend, abstract_backend.AbstractBackend):
    backend_name = backend.name
  else:
    backend_name = backend
  for arg, val in kwargs.items():
    if arg not in CONFIG_ARGS[backend]:
      raise ValueError(f"unkown config-variable {arg} "
                       f"for {backend_name} backend")
    CONFIG_ARGS[backend_name][arg] = val  #TODO(mganahl): add check for val

  if isinstance(backend, abstract_backend.AbstractBackend):
    backend_name = backend.name
  else:
    backend_name = backend
  # the passed backend does not have to be in _INSTANTIATED_BACKENDS
  if isinstance(backend, abstract_backend.AbstractBackend):
    backend.configure(**CONFIG_ARGS[backend_name])
    
  backend_names = _INSTANTIATED_BACKENDS.keys()
  if backend_name in backend_names:
    # if this backend is already in use, reconfigure it
    _INSTANTIATED_BACKENDS[backend_name].configure(**CONFIG_ARGS[backend_name])
    
def reset_backend(backend: Union[Text, abstract_backend.AbstractBackend])->None:
  """
  Reset `backend` to its default configuration.
  """
  if isinstance(backend, abstract_backend.AbstractBackend):
    backend.config(**CONFIG_ARGS[backend])    
    backend_name = backend.name
  else:
    backend_name = backend
  backend_names = _INSTANTIATED_BACKENDS.keys()    
  if backend_name in backend_names:
    _INSTANTIATED_BACKENDS[backend_name].configure(**CONFIG_ARGS[backend])
    
  
def get_backend(
    backend: Union[Text, abstract_backend.AbstractBackend]
) -> abstract_backend.AbstractBackend:
  
  if isinstance(backend, abstract_backend.AbstractBackend):
    backend = backend.name
  if backend not in _BACKENDS:
    raise ValueError("Backend '{}' does not exist".format(backend))
  if backend in _INSTANTIATED_BACKENDS:
    return  _INSTANTIATED_BACKENDS[backend]
  
  _INSTANTIATED_BACKENDS[backend] = _BACKENDS[backend](**CONFIG_ARGS[backend])
  return _INSTANTIATED_BACKENDS[backend]
