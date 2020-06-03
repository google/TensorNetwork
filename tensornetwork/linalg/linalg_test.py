import numpy as np
import time
import pytest
import jax.numpy as jnp
import jax.config as config
import torch
import tensorflow as tf
from tensornetwork.linalg import linalg
from tensornetwork.network_components import Node
from tensornetwork.backend_contextmanager import DefaultBackend
from tensornetwork import backends
from tensornetwork.backends.numpy import numpy_backend
from tensornetwork.backends.jax import jax_backend
#pylint: disable=no-member
config.update("jax_enable_x64", True)

np_real = [np.float32, np.float16, np.float64]
np_float = np_real + [np.complex64, np.complex128]
np_int = [np.int8, np.int16, np.int32, np.int64]
np_uint = [np.uint8, np.uint16, np.uint32, np.uint64]
np_dtypes = {
    "real": np_real,
    "float": np_float,
    "rand": np_float,
    "int": np_int + np_uint,
    "all": np_real + np_int + np_uint + [
        None,
    ]
}

tf_real = [tf.float32, tf.float16, tf.float64]
tf_float = tf_real + [tf.complex64, tf.complex128]
tf_int = [tf.int8, tf.int16, tf.int32, tf.int64]
tf_uint = [tf.uint8, tf.uint16, tf.uint32, tf.uint64]
tf_dtypes = {
    "real": tf_real,
    "float": tf_float,
    "rand": tf_real + [
        None,
    ],
    "int": tf_int + tf_uint,
    "all": tf_real + tf_int + tf_uint + [
        None,
    ]
}

torch_float = [torch.float32, torch.float16, torch.float64]
torch_int = [torch.int8, torch.int16, torch.int32, torch.int64]
torch_uint = [torch.uint8]
torch_dtypes = {
    "real": torch_float,
    "float": torch_float,
    "rand": [torch.float32, torch.float64, None],
    "int": torch_int + torch_uint,
    "all": torch_float + torch_int + torch_uint + [
        None,
    ]
}

dtypes = {
    "pytorch": torch_dtypes,
    "jax": np_dtypes,
    "numpy": np_dtypes,
    "tensorflow": tf_dtypes
}


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
    tnI = linalg.eye(
        N, dtype=dtype, M=M, name=name, axis_names=axis_names, backend=backend)
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
    tnI = linalg.zeros(
        shape, dtype=dtype, name=name, axis_names=axis_names, backend=backend)
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
    tnI = linalg.ones(
        shape, dtype=dtype, name=name, axis_names=axis_names, backend=backend)
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
    tnI = linalg.randn(
        shape,
        dtype=dtype,
        name=name,
        axis_names=axis_names,
        backend=backend,
        seed=seed)
    npI = backend_obj.randn(shape, dtype=dtype, seed=seed)
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
    tnI = linalg.random_uniform(
        shape,
        dtype=dtype,
        name=name,
        axis_names=axis_names,
        backend=backend,
        seed=seed,
        boundaries=boundaries)
    npI = backend_obj.random_uniform(
        shape, dtype=dtype, seed=seed, boundaries=boundaries)
    np.testing.assert_allclose(tnI.tensor, npI)
    assert tnI.name == name
    edges = tnI.get_all_dangling()
    for edge, expected_name in zip(edges, axis_names):
      assert edge.name == expected_name
    assert tnI.backend.name == backend


def test_conj(backend):
  if backend == "pytorch":
    pytest.skip("Complex numbers currently not supported in PyTorch")

  a = Node(np.random.rand(3, 3) + 1j * np.random.rand(3, 3), backend=backend)
  abar = linalg.conj(a)
  np.testing.assert_allclose(abar.tensor, a.backend.conj(a.tensor))


def test_transpose(backend):
  a = Node(np.random.rand(1, 2, 3, 4, 5), backend=backend)
  order = [a[n] for n in reversed(range(5))]
  transpa = linalg.transpose(a, [4, 3, 2, 1, 0])
  a.reorder_edges(order)
  np.testing.assert_allclose(a.tensor, transpa.tensor)


def test_operator_kron(backend):
  with DefaultBackend(backend):
    X = np.array([[0, 1], [1, 0]], dtype=np.float32)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float32)
    expected = np.kron(X, Z).reshape(2, 2, 2, 2)
    result = linalg.kron([Node(X), Node(Z)])
    np.testing.assert_allclose(result.tensor, expected)


def test_kron_raises(backend):
  with DefaultBackend(backend):
    A = Node(np.ones((2, 2, 2)))
    B = Node(np.ones((2, 2, 2)))
    with pytest.raises(
        ValueError, match="All operator tensors must have an even order."):
      linalg.kron([A, B])


def test_norm_of_node_without_backend_raises_error():
  node = np.random.rand(3, 3, 3)
  with pytest.raises(AttributeError):
    linalg.norm(node)


def test_conj_of_node_without_backend_raises_error():
  node = np.random.rand(3, 3, 3)
  with pytest.raises(AttributeError):
    linalg.conj(node)


def test_transpose_of_node_without_backend_raises_error():
  node = np.random.rand(3, 3, 3)
  with pytest.raises(AttributeError):
    linalg.transpose(node, permutation=[])
