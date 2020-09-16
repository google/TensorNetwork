from typing import Text, Union
from tensornetwork.backends.abstract_backend import AbstractBackend
from tensornetwork.backends import backend_factory


class DefaultBackend():
  """Context manager for setting up backend for nodes"""

  def __init__(self, backend: Union[Text, AbstractBackend]) -> None:
    if not isinstance(backend, (Text, AbstractBackend)):
      raise ValueError("Item passed to DefaultBackend "
                       "must be Text or BaseBackend")
    self.backend = backend

  def __enter__(self):
    _default_backend_stack.stack.append(self)

  def __exit__(self, exc_type, exc_val, exc_tb):
    _default_backend_stack.stack.pop()


class _DefaultBackendStack():
  """A stack to keep track default backends context manager"""

  def __init__(self):
    self.stack = []
    self.default_backend = "numpy"

  def get_current_backend(self):
    return self.stack[-1].backend if self.stack else self.default_backend


_default_backend_stack = _DefaultBackendStack()


def get_default_backend():
  return _default_backend_stack.get_current_backend()


def set_default_backend(backend: Union[Text, AbstractBackend]) -> None:
  if _default_backend_stack.stack:
    raise AssertionError("The default backend should not be changed "
                         "inside the backend context manager")
  if not isinstance(backend, (Text, AbstractBackend)):
    raise ValueError("Item passed to set_default_backend "
                     "must be Text or BaseBackend")
  if isinstance(backend, Text) and backend not in backend_factory._BACKENDS:
    raise ValueError(f"Backend '{backend}' was not found.")
  _default_backend_stack.default_backend = backend
