import tensornetwork as tn
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
    assert tn.config.default_backend == "pytorch"

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
  try:
    with tn.DefaultBackend(a):
      pass
    assert False
  except ValueError:
    assert True

def test_contextmanager_BaseBackend():
  a = tn.Node(np.ones((10,)))
  tn.set_default_backend("pytorch")
  with tn.DefaultBackend(a.backend):
    b = tn.Node(np.ones((10,)))
  assert b.backend.name == "numpy"
