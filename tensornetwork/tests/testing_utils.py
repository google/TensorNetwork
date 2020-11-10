import numpy as np
import jax.numpy as jnp
from jax import config
import tensorflow as tf
import torch
import pytest
import tensornetwork
from tensornetwork.block_sparse.charge import charge_equal
from tensornetwork import backends
config.update("jax_enable_x64", True)

np_real = [np.float32, np.float64]
np_complex = [np.complex64, np.complex128]
np_float_dtypes = np_real + np_complex
np_int = [np.int8, np.int16, np.int32, np.int64]
np_uint = [np.uint8, np.uint16, np.uint32, np.uint64]
np_not_bool = np_float_dtypes + np_int + np_uint + [None, ]
np_not_half = [np.float32, np.float64] + np_complex
np_all_dtypes = np_not_bool + [np.bool, ]

torch_supported_dtypes = np_real + np_int + [np.uint8, np.bool, None]
# torch_supported_dtypes = [np.float32, np.float64]


def safe_randn(shape, backend, dtype):
  """
  Creates a random tensor , catching errors that occur when the
  dtype is not supported by the backend. Returns the Tensor and the backend
  array, which are both None if the dtype and backend did not match.
  """
  np.random.seed(seed=10)
  init = np.random.randn(*shape)
  if dtype == np.bool:
    init = np.round(init)
  init = init.astype(dtype)

  if dtype in np_complex:
    init_i = np.random.randn(*shape)
    init = init + 1.0j * init_i.astype(dtype)

  if backend == "pytorch" and dtype not in torch_supported_dtypes:
    pytest.skip("dtype unsupported by PyTorch")
  else:
    A = tensornetwork.Tensor(init, backend=backend)
  return (A, init)


def safe_zeros(shape, backend, dtype):
  """
  Creates a tensor of zeros, catching errors that occur when the
  dtype is not supported by the backend. Returns both the Tensor and the backend
  array, which are both None if the dtype and backend did not match.
  """
  init = np.zeros(shape, dtype=dtype)
  if backend == "pytorch" and dtype not in torch_supported_dtypes:
    pytest.skip("dtype unsupported by PyTorch")
  else:
    A = tensornetwork.Tensor(init, backend=backend)
  return (A, init)


def np_dtype_to_backend(backend, dtype):
  """
  Converts a given np dtype to the equivalent in the given backend. Skips
  the present test if the dtype is not supported in the backend.
  """
  backend_obj = backends.backend_factory.get_backend(backend)
  if backend_obj.name in ("numpy", "symmetric"):
    return dtype
  A_np = np.ones([1], dtype=dtype)

  if backend_obj.name == "jax":
    A = jnp.array(A_np)
  elif backend_obj.name == "tensorflow":
    A = tf.convert_to_tensor(A_np, dtype=dtype)
  elif backend_obj.name == "pytorch":
    if dtype not in torch_supported_dtypes:
      pytest.skip("dtype unsupported by PyTorch")
    A = torch.tensor(A_np)
  else:
    raise ValueError("Invalid backend ", backend)
  return A.dtype


def check_contraction_dtype(backend, dtype):
  """
  Skips the test if the backend cannot perform multiply-add with the given
  dtype.
  """
  skip = False
  if backend == "tensorflow":
    if dtype in [np.uint8, tf.uint8, np.uint16, tf.uint16, np.int8, tf.int8,
                 np.int16, tf.int16, np.uint32, tf.uint32, np.uint64,
                 tf.uint64]:
      skip = True

  if backend == "pytorch":
    if dtype in [np.float16, torch.float16]:
      skip = True
  if skip:
    pytest.skip("backend does not support multiply-add with this dtype.")


def assert_allclose(expected, actual, backend, **kwargs):
  if backend.name == 'symmetric':
    exp = expected.contiguous()
    act = actual.contiguous()
    if exp.shape != act.shape:
      raise ValueError(f"expected shape = {exp.shape}, "
                       f"actual shape = {act.shape}")
    if len(exp.flat_charges) != len(act.flat_charges):
      raise ValueError("expected charges differ from actual charges")

    if len(exp.flat_flows) != len(act.flat_flows):
      raise ValueError(f"expected flat flows = {exp.flat_flows}"
                       f" differ from actual flat flows = {act.flat_flows}")

    for c1, c2 in zip(exp.flat_charges, act.flat_charges):
      if not charge_equal(c1, c2):
        raise ValueError("expected charges differ from actual charges")

    if not np.all(np.array(exp.flat_flows) == np.array(act.flat_flows)):
      raise ValueError(f"expected flat flows = {exp.flat_flows}"
                       f" differ from actual flat flows = {act.flat_flows}")
    if not np.all(np.abs(exp.data - act.data) < 1E-10):
      np.testing.assert_allclose(act.data, exp.data, **kwargs)
  else:
    np.testing.assert_allclose(actual, expected, **kwargs)
