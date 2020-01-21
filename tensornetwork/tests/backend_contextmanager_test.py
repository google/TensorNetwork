import tensornetwork as tn
import pytest
import numpy as np

def test_contextmanager_simple():
  tn.set_default_backend("pytorch")
  with tn.DefaultBackend("numpy"):
    a = tn.Node(np.ones((10,)))
    b = tn.Node(np.ones((10,)))
  assert a.backend.name == b.backend.name
  edge = a[0] ^ b[0]
  final_node = tn.contract(edge)
  assert final_node.tensor == 10.0


def test_contextmanager_interaption():
  tn.set_default_backend("pytorch")
  assert tn.config.default_backend == "pytorch"
  with tn.DefaultBackend("numpy"):
    assert tn.config.default_backend == "pytorch"
    a = tn.Node(np.ones((10,)))
    tn.set_default_backend("tensorflow")
    b = tn.Node(np.ones((10,)))
  assert a.backend.name == b.backend.name
  assert tn.config.default_backend == "tensorflow"

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
