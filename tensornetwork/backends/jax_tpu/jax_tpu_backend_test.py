import jax
import numpy as np
from tensornetwork.backends.jax_tpu import jax_tpu_backend
from tensornetwork.backends.shell import shell_backend
import pytest
import tensornetwork


def test_jax_tpu_tensor_sanity_check():
  tmp_tensor = np.ones((8, 32, 64))
  tpu_tensor = jax_tpu_backend.JaxTPUTensor(
      tmp_tensor, shell_backend.ShellTensor(tmp_tensor.shape))
  assert tpu_tensor.shape == (8, 32, 64)
  assert tpu_tensor.concrete_tensor.shape == (8, 2048)
  assert isinstance(tpu_tensor.concrete_tensor, jax.numpy.ndarray)


def test_jax_tpu_tensor_small_shape():
  tmp_tensor = np.ones((8, 8, 8))
  tpu_tensor = jax_tpu_backend.JaxTPUTensor(
      tmp_tensor, shell_backend.ShellTensor(tmp_tensor.shape))
  assert tpu_tensor.shape == (8, 8, 8)
  assert tpu_tensor.concrete_tensor.shape == (1, 512)
  assert isinstance(tpu_tensor.concrete_tensor, jax.numpy.ndarray)


def test_tensordot():
  tmp_tensor = np.ones((8, 32, 64))
  backend = jax_tpu_backend.JaxTPUBackend()
  tpu_tensor1 = jax_tpu_backend.JaxTPUTensor(
      tmp_tensor, shell_backend.ShellTensor(tmp_tensor.shape))
  tpu_tensor2 = jax_tpu_backend.JaxTPUTensor(
      tmp_tensor, shell_backend.ShellTensor(tmp_tensor.shape))
  result = backend.tensordot(tpu_tensor1, tpu_tensor2, [[0], [0]])
  assert result.shape == (32, 64, 32, 64)
  assert result.concrete_tensor.shape == (2048, 2048)


def test_tensordot_no_optimization():
  tmp_tensor = np.ones((8, 32, 64))
  backend = jax_tpu_backend.JaxTPUBackend()
  tpu_tensor1 = jax_tpu_backend.JaxTPUTensor(
      tmp_tensor, shell_backend.ShellTensor(tmp_tensor.shape))
  tpu_tensor2 = jax_tpu_backend.JaxTPUTensor(
      tmp_tensor, shell_backend.ShellTensor(tmp_tensor.shape))
  result = backend.tensordot(tpu_tensor1, tpu_tensor2, [[2], [2]])
  assert result.shape == (8, 32, 8, 32)
  assert result.concrete_tensor.shape == (256, 256)


def test_tensordot_multi_axes():
  tmp_tensor = np.ones((8, 32, 64))
  backend = jax_tpu_backend.JaxTPUBackend()
  tpu_tensor1 = jax_tpu_backend.JaxTPUTensor(
      tmp_tensor, shell_backend.ShellTensor(tmp_tensor.shape))
  tpu_tensor2 = jax_tpu_backend.JaxTPUTensor(
      tmp_tensor, shell_backend.ShellTensor(tmp_tensor.shape))
  result = backend.tensordot(tpu_tensor1, tpu_tensor2, [[0, 1], [0, 1]])
  assert result.shape == (64, 64)
  assert result.concrete_tensor.shape == (1, 4096)


def test_tensornetwork_integration():
  net = tensornetwork.TensorNetwork("jax_tpu")
  a = net.add_node(np.ones((32, 32, 128)))
  b = net.add_node(np.ones((32, 32, 128)))
  c = net.add_node(np.ones((32, 32, 128)))
  # pylint: disable=pointless-statement
  a[0] ^ b[1]
  b[0] ^ c[1]
  c[0] ^ a[1]
  d = a @ b
  e = d @ c
  assert e.shape == (128, 128, 128)
  assert e.tensor.concrete_tensor.shape == (16384, 128)
  assert e.tensor.real_tensor.shape == (128, 128, 128)


def test_flatten_all_edges():
  net = tensornetwork.TensorNetwork("jax_tpu")
  a = net.add_node(np.ones((32, 32, 16)))
  b = net.add_node(np.ones((32, 32, 16)))
  c = net.add_node(np.ones((32, 32, 16)))
  d = net.add_node(np.ones((16, 32, 32)))
  # pylint: disable=pointless-statement
  a[0] ^ b[0]
  a[1] ^ b[1]
  c[0] ^ d[1]
  c[1] ^ d[2]
  c[2] ^ a[2]
  net.flatten_all_edges()
  final_node = tensornetwork.contractors.naive(net).get_final_node()
  assert final_node.tensor.real_tensor.shape == (16, 16)
