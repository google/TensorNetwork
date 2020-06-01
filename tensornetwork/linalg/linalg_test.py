import numpy as np
import time
import pytest
import jax.numpy as jnp
import jax.config as config
import torch
import tensorflow as tf
from tensornetwork.linalg import linalg
from tensornetwork import backends
from tensornetwork.backends.numpy import numpy_backend
from tensornetwork.backends.jax import jax_backend
#pylint: disable=no-member
config.update("jax_enable_x64", True)


np_real = [np.float32, np.float16, np.float64]
np_float = np_real + [np.complex64, np.complex128]
np_int = [np.int8, np.int16, np.int32, np.int64]
np_uint = [np.uint8, np.uint16, np.uint32, np.uint64]
np_dtypes = {"real": np_real, "float": np_float,
             "rand": np_float,
             "int": np_int + np_uint,
             "all": np_real+ np_int + np_uint}

tf_real = [tf.float32, tf.float16, tf.float64]
tf_float = tf_real + [tf.complex64, tf.complex128]
tf_int = [tf.int8, tf.int16, tf.int32, tf.int64]
tf_uint = [tf.uint8, tf.uint16, tf.uint32, tf.uint64]
tf_dtypes = {"real": tf_real, "float": tf_float,
             "rand": tf_real,
             "int": tf_int + tf_uint,
             "all": tf_real + tf_int + tf_uint}

torch_float = [torch.float32, torch.float16, torch.float64]
torch_int = [torch.int8, torch.int16, torch.int32, torch.int64]
torch_uint = [torch.uint8]
torch_dtypes = {"real": torch_float, "float": torch_float,
                "rand": [torch.float32, torch.float64],
                "int": torch_int + torch_uint,
                "all": torch_float + torch_int + torch_uint}

dtypes = {"pytorch": torch_dtypes,
          "jax": np_dtypes, "numpy": np_dtypes, "tensorflow": tf_dtypes}


def test_eye(backend):
  """
  Tests linalg.eye against np.eye.
  """
  N = 4
  M = 6
  name = "Jeffrey"
  axis_names = ["Sam", "Blinkey"]
  backend_obj = backends.backend_factory.get_backend(backend)
  for dtype in dtypes[backend]["all"]:
    tnI = linalg.eye(N, dtype=dtype, M=M, name=name, axis_names=axis_names,
                     backend=backend)
    npI = backend_obj.eye(N, dtype=dtype, M=M)
    np.testing.assert_allclose(tnI.tensor, npI)
    assert tnI.name == name
    edges = tnI.get_all_dangling()
    for edge, expected_name in zip(edges, axis_names):
      assert edge.name == expected_name
    assert tnI.backend.name == backend


def test_zeros(backend):
  """
  Tests linalg.zeros against np.zeros.
  """
  shape = (5, 10, 3)
  name = "Jeffrey"
  axis_names = ["Sam", "Blinkey", "Renaldo"]
  backend_obj = backends.backend_factory.get_backend(backend)
  for dtype in dtypes[backend]["all"]:
    tnI = linalg.zeros(shape, dtype=dtype, name=name, axis_names=axis_names,
                       backend=backend)
    npI = backend_obj.zeros(shape, dtype=dtype)
    np.testing.assert_allclose(tnI.tensor, npI)
    assert tnI.name == name
    edges = tnI.get_all_dangling()
    for edge, expected_name in zip(edges, axis_names):
      assert edge.name == expected_name
    assert tnI.backend.name == backend


def test_ones(backend):
  """
  Tests linalg.ones against np.ones.
  """
  shape = (5, 10, 3)
  name = "Jeffrey"
  axis_names = ["Sam", "Blinkey", "Renaldo"]
  backend_obj = backends.backend_factory.get_backend(backend)
  for dtype in dtypes[backend]["all"]:
    tnI = linalg.ones(shape, dtype=dtype, name=name, axis_names=axis_names,
                      backend=backend)
    npI = backend_obj.ones(shape, dtype=dtype)
    np.testing.assert_allclose(tnI.tensor, npI)
    assert tnI.name == name
    edges = tnI.get_all_dangling()
    for edge, expected_name in zip(edges, axis_names):
      assert edge.name == expected_name
    assert tnI.backend.name == backend


def test_randn(backend):
  """
  Tests linalg.randn against the backend code.
  """
  shape = (5, 10, 3, 2)
  seed = int(time.time())
  np.random.seed(seed=seed)
  name = "Jeffrey"
  axis_names = ["Sam", "Blinkey", "Renaldo", "Jarvis"]
  backend_obj = backends.backend_factory.get_backend(backend)
  for dtype in dtypes[backend]["rand"]:
    tnI = linalg.randn(shape, dtype=dtype, name=name, axis_names=axis_names,
                       backend=backend, seed=seed)
    npI = backend_obj.randn(shape, dtype=dtype, seed=seed)
    data = tnI.tensor
    np.testing.assert_allclose(tnI.tensor, npI)
    assert tnI.name == name
    edges = tnI.get_all_dangling()
    for edge, expected_name in zip(edges, axis_names):
      assert edge.name == expected_name
    assert tnI.backend.name == backend


def test_random_uniform(backend):
  """
  Tests linalg.ones against np.ones.
  """
  shape = (5, 10, 3, 2)
  seed = int(time.time())
  np.random.seed(seed=seed)
  boundaries = (-0.3, 10.5)
  name = "Jeffrey"
  axis_names = ["Sam", "Blinkey", "Renaldo", "Jarvis"]
  backend_obj = backends.backend_factory.get_backend(backend)
  for dtype in dtypes[backend]["rand"]:
    tnI = linalg.random_uniform(shape, dtype=dtype, name=name,
                                axis_names=axis_names, backend=backend,
                                seed=seed, boundaries=boundaries)
    npI = backend_obj.random_uniform(shape, dtype=dtype, seed=seed,
                                     boundaries=boundaries)
    np.testing.assert_allclose(tnI.tensor, npI)
    assert tnI.name == name
    edges = tnI.get_all_dangling()
    for edge, expected_name in zip(edges, axis_names):
      assert edge.name == expected_name
    assert tnI.backend.name == backend
