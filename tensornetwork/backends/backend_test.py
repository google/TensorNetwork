"""Tests for graphmode_tensornetwork."""
import builtins
import sys
import pytest
import numpy as np


def clean_tensornetwork_modules():
  for mod in list(sys.modules.keys()):
    if mod.startswith('tensornetwork'):
      sys.modules.pop(mod, None)


@pytest.fixture(autouse=True)
def clean_backend_import():
  #never do this outside testing
  clean_tensornetwork_modules()
  yield # use as teardown
  clean_tensornetwork_modules()


@pytest.fixture
def no_backend_dependency(monkeypatch):
  import_orig = builtins.__import__
  # pylint: disable=redefined-builtin
  def mocked_import(name, globals, locals, fromlist, level):
    if name in ['torch', 'tensorflow', 'jax']:
      raise ImportError()
    return import_orig(name, globals, locals, fromlist, level)
  monkeypatch.setattr(builtins, '__import__', mocked_import)


@pytest.mark.usefixtures('no_backend_dependency')
def test_backend_pytorch_missing_cannot_initialize_backend():
  with pytest.raises(ImportError):
    from tensornetwork.backends.pytorch.pytorch_backend import PyTorchBackend
    PyTorchBackend()


@pytest.mark.usefixtures('no_backend_dependency')
def test_backend_tensorflow_missing_cannot_initialize_backend():
  with pytest.raises(ImportError):
    from tensornetwork.backends.tensorflow.tensorflow_backend \
      import TensorFlowBackend
    TensorFlowBackend()


@pytest.mark.usefixtures('no_backend_dependency')
def test_backend_jax_missing_cannot_initialize_backend():
  with pytest.raises(ImportError):
    from tensornetwork.backends.jax.jax_backend import JaxBackend
    JaxBackend()


@pytest.mark.usefixtures('no_backend_dependency')
def test_config_backend_missing_can_import_config():
  import tensornetwork.config
  with pytest.raises(ImportError):
    import torch
  with pytest.raises(ImportError):
    import tensorflow as tf
  with pytest.raises(ImportError):
    import jax


@pytest.mark.usefixtures('no_backend_dependency')
def test_import_tensornetwork_without_backends():
  import tensornetwork
  import tensornetwork.backends.pytorch.pytorch_backend
  import tensornetwork.backends.tensorflow.tensorflow_backend
  import tensornetwork.backends.jax.jax_backend
  import tensornetwork.backends.numpy.numpy_backend
  with pytest.raises(ImportError):
    import torch
  with pytest.raises(ImportError):
    import tensorflow as tf
  with pytest.raises(ImportError):
    import jax


@pytest.mark.usefixtures('no_backend_dependency')
def test_basic_numpy_network_without_backends():
  import tensornetwork
  net = tensornetwork.TensorNetwork(backend="numpy")
  a = net.add_node(np.ones((10,)))
  b = net.add_node(np.ones((10,)))
  edge = net.connect(a[0], b[0])
  final_node = net.contract(edge)
  assert final_node.tensor == np.array(10.)
  with pytest.raises(ImportError):
    import torch
  with pytest.raises(ImportError):
    import tensorflow as tf
  with pytest.raises(ImportError):
    import jax
