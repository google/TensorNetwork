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

"""
Decorator functions that depend on the backend.
"""
from typing import Union, Iterable, Optional, Text, Callable
import functools
import tensornetwork.backends.abstract_backend as abstract_backend
import tensornetwork.backends as backends
import tensornetwork.backend_contextmanager as backend_contextmanager

AbstractBackend = abstract_backend.AbstractBackend

def jit(fun: Callable,
        backend: Union[Text, AbstractBackend] = None,
        backend_argnum: Optional[int] = None,
        static_argnums: Union[int, Iterable[int]] = (), device=None,
        xla_backend: Optional[str] = None) -> Callable:
  """
  Return a jitted or graph-compiled version of `fun`
  for JAX backend. For all other backends returns `fun`.
  Args:
    fun: Callable
    backend: The backend.
    backend_argnum: Labels the argument of the decorated function which
                    specifies the backend.
                    This argument will be treated
                    as static in the sense of static_argnums.
                    If backend_argnum is specified, backend must be None.
    static_argnums: Label the arguments which will be statically compiled
                    against.
    xla_backend: Specifies the backend ('gpu', 'cpu'...) against which
                 XLA is to run.
    donate_argnums: Labels arguments that Jit is allowed to overwrite.
    args: Arguments to `fun`.
    kwargs: Keyword arguments to `fun`.

  Raises:
    ValueError: If backend_argnum is specified but backend is not None.

                If backend_argnum is specified but the corresponding
                argument neither is nor labels a backend.
  Returns:
    Callable: jitted/graph-compiled version of `fun`, or just `fun`.
  """
  argnum_mode = False
  if backend_argnum is not None:
    if backend is not None:
      raise ValueError("backend must be None if backend_argnum is specified.")
    argnum_mode = True
    static_argnums = tuple(list(static_argnums) + [backend_argnum,])

  if not argnum_mode:
    if backend is None:
      backend = backend_contextmanager.get_default_backend()
    backend_obj = backends.backend_factory.get_backend(backend)

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
      jitted = backend_obj.jit(fun, static_argnums=static_argnums,
                               device=device, backend=xla_backend)
      return jitted(*args, **kwargs)
  else:
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
      backend = args[backend_argnum]
      try:
        backend_obj = backends.backend_factory.get_backend(backend)
      except ValueError as error:
        errstr = (f"backend_argnum={backend_argnum} was specified"
                  f"but the corresponding argument {args[backend_argnum]}"
                  f"did not specify a backend.")
        raise ValueError(errstr) from error
      jitted = backend_obj.jit(fun, static_argnums=static_argnums,
                               device=device, backend=xla_backend)
      return jitted(*args, **kwargs)
  return wrapper
