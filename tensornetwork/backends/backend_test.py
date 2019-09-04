"""Tests for graphmode_tensornetwork."""
import builtins
import sys
import pytest


@pytest.fixture(autouse=True)
def clean_backend_import():
  #never do this outside testing
  sys.modules.pop('tensornetwork.backends.pytorch.pytorch_backend', None)
  sys.modules.pop('tensornetwork.backends.jax.jax_backend', None)
  sys.modules.pop('tensornetwork.backends.tensorflow.tensorflow_backend', None)
  sys.modules.pop('tensornetwork', None)
  sys.modules.pop('tensornetwork.network', None)
  sys.modules.pop('tensornetwork.network_components', None)
  yield # use as teardown
  sys.modules.pop('tensornetwork', None)


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
def test_backend_pytorch_missing():
  with pytest.raises(ImportError):
    from tensornetwork.backends.pytorch.pytorch_backend import PyTorchBackend


@pytest.mark.usefixtures('no_backend_dependency')
def test_backend_tensorflow_missing():
  with pytest.raises(ImportError):
    import tensornetwork.backends.tensorflow.tensorflow_backend


@pytest.mark.usefixtures('no_backend_dependency')
def test_backend_jax_missing():
  with pytest.raises(ImportError):
    from tensornetwork.backends.jax.jax_backend import JaxBackend


@pytest.mark.usefixtures('no_backend_dependency')
def test_config_pytorch_missing():
  import tensornetwork.config
  with pytest.raises(ImportError):
    import torch


@pytest.mark.usefixtures('no_backend_dependency')
def test_config_tensorflow_missing():
  import tensornetwork.config
  with pytest.raises(ImportError):
    import tensorflow as tf


@pytest.mark.usefixtures('no_backend_dependency')
def test_import_tensornetwork_withoutbackends():
  import tensornetwork
  import tensornetwork.network
  import tensornetwork.network_components
