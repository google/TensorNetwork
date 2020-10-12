import numpy as np
import time
import pytest
import jax.numpy as jnp
import jax.config as config
import torch
import tensorflow as tf
import tensornetwork
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
  Tests tensornetwork.eye against np.eye.
  """
  N = 4
  M = 6
  backend_obj = backends.backend_factory.get_backend(backend)
  for dtype in dtypes[backend]["all"]:
    tnI = tensornetwork.eye(N, dtype=dtype, M=M, backend=backend)
    npI = backend_obj.eye(N, dtype=dtype, M=M)
    np.testing.assert_allclose(tnI.array, npI)


def test_zeros(backend):
  """
  Tests tensornetwork.zeros against np.zeros.
  """
  shape = (5, 10, 3)
  backend_obj = backends.backend_factory.get_backend(backend)
  for dtype in dtypes[backend]["all"]:
    tnI = tensornetwork.zeros(shape, dtype=dtype, backend=backend)
    npI = backend_obj.zeros(shape, dtype=dtype)
    np.testing.assert_allclose(tnI.array, npI)


def test_ones(backend):
  """
  Tests tensornetwork.ones against np.ones.
  """
  shape = (5, 10, 3)
  backend_obj = backends.backend_factory.get_backend(backend)
  for dtype in dtypes[backend]["all"]:
    tnI = tensornetwork.ones(shape, dtype=dtype, backend=backend)
    npI = backend_obj.ones(shape, dtype=dtype)
    np.testing.assert_allclose(tnI.array, npI)


def test_randn(backend):
  """
  Tests tensornetwork.randn against the backend code.
  """
  shape = (5, 10, 3, 2)
  seed = int(time.time())
  np.random.seed(seed=seed)
  backend_obj = backends.backend_factory.get_backend(backend)
  for dtype in dtypes[backend]["rand"]:
    tnI = tensornetwork.randn(
        shape,
        dtype=dtype,
        seed=seed,
        backend=backend)
    npI = backend_obj.randn(shape, dtype=dtype, seed=seed)
    np.testing.assert_allclose(tnI.array, npI)


def test_random_uniform(backend):
  """
  Tests tensornetwork.ones against np.ones.
  """
  shape = (5, 10, 3, 2)
  seed = int(time.time())
  np.random.seed(seed=seed)
  boundaries = (-0.3, 10.5)
  backend_obj = backends.backend_factory.get_backend(backend)
  for dtype in dtypes[backend]["rand"]:
    tnI = tensornetwork.random_uniform(
        shape,
        dtype=dtype,
        seed=seed,
        boundaries=boundaries,
        backend=backend)
    npI = backend_obj.random_uniform(
        shape, dtype=dtype, seed=seed, boundaries=boundaries)
    np.testing.assert_allclose(tnI.array, npI)


@pytest.mark.parametrize("shape", (2, 4, 1))
@pytest.mark.parametrize("n", np.eye(2))
def test_ones_like(backend, shape, n):
    """Tests tensornetwork.ones_like against np.zeros_like"""
    backend_obj = backends.backend_factory.get_backend(backend)

    @pytest.mark.parametrize("dtype,expected", (dtypes[backend]["all"]))
    def inner_ones_test(dtype):
        objTensor = tensornetwork.ones(shape, dtype=dtype,
                                       backend=backend)
        tensor = tensornetwork.ones_like(objTensor, dtype=dtype,
                                         backend=backend)
        numpyT = tensornetwork.ones_like(n, dtype=dtype,
                                         backend=backend)
        tensorCheck = backend_obj.ones(shape, dtype=dtype)
        numpyCheck = backend_obj.ones(n.shape, dtype=dtype)
        np.testing.assert_allclose(tensor.array, tensorCheck)
        np.testing.assert_allclose(numpyT.array, numpyCheck)


@pytest.mark.parametrize("shape", (2, 4, 1))
@pytest.mark.parametrize("n", np.eye(2))
def test_zeros_like(backend, shape, n):
    """Tests tensornetwork.zeros_like against np.zeros_like"""
    backend_obj = backends.backend_factory.get_backend(backend)

    @pytest.mark.parametrize("dtype,expected", (dtypes[backend]["all"]))
    def inner_zero_test(dtype):
        objTensor = tensornetwork.zeros(shape, dtype=dtype, backend=backend)
        tensor = tensornetwork.zeros_like(objTensor, dtype=dtype,
                                          backend=backend)
        numpyT = tensornetwork.zeros_like(n, dtype=dtype,
                                          backend=backend)
        tensorCheck = backend_obj.zeros(shape, dtype=dtype)
        numpyCheck = backend_obj.zeros(n.shape, dtype=dtype)
        np.testing.assert_allclose(tensor.array, tensorCheck)
        np.testing.assert_allclose(numpyT.array, numpyCheck)
