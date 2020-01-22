from typing import Text, Union
from tensornetwork.backends.base_backend import BaseBackend
import tensornetwork.config as config

class DefaultBackend():
  """Context manager for setting up backend for nodes"""

  def __init__(self, backend: Union[Text, BaseBackend]) -> None:
    if not isinstance(backend, (Text, BaseBackend)):
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

  def get_current_backend(self):
    return self.stack[-1].backend if self.stack else config.default_backend

_default_backend_stack = _DefaultBackendStack()

def get_default_backend():
  return _default_backend_stack.get_current_backend()
