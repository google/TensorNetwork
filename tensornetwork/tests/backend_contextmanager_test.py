import tensornetwork as tn
from tensornetwork.backend_contextmanager import _default_backend_stack
import pytest
import numpy as np


def test_contextmanager_simple():
  with tn.DefaultBackend("tensorflow"):
    a = tn.Node(np.ones((10,)))
    b = tn.Node(np.ones((10,)))
  assert a.backend.name == b.backend.name


def test_contextmanager_default_backend():
  tn.set_default_backend("pytorch")
  with tn.DefaultBackend("numpy"):
    assert _default_backend_stack.default_backend == "pytorch"


def test_contextmanager_interruption():
  tn.set_default_backend("pytorch")
  with pytest.raises(AssertionError):
    with tn.DefaultBackend("numpy"):
      tn.set_default_backend("tensorflow")


def test_contextmanager_nested():
  with tn.DefaultBackend("tensorflow"):
    a = tn.Node(np.ones((10,)))
    assert a.backend.name == "tensorflow"
    with tn.DefaultBackend("numpy"):
      b = tn.Node(np.ones((10,)))
      assert b.backend.name == "numpy"
    c = tn.Node(np.ones((10,)))
    assert c.backend.name == "tensorflow"
  d = tn.Node(np.ones((10,)))
  assert d.backend.name == "numpy"


def test_contextmanager_wrong_item():
  a = tn.Node(np.ones((10,)))
  with pytest.raises(ValueError):
    tn.DefaultBackend(a)  # pytype: disable=wrong-arg-types



def test_contextmanager_BaseBackend():
  tn.set_default_backend("pytorch")
  a = tn.Node(np.ones((10,)))
  with tn.DefaultBackend(a.backend):
    b = tn.Node(np.ones((10,)))
  assert b.backend.name == "pytorch"


def test_set_default_backend_value_error():
  tn.set_default_backend("pytorch")
  with pytest.raises(
      ValueError,
      match="Item passed to set_default_backend "
      "must be Text or BaseBackend"):
    tn.set_default_backend(-1)  # pytype: disable=wrong-arg-types
