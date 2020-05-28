import numpy as np
import jax.numpy as jnp
import jax.config as config
from tensornetwork.linalg import linalg
from tensornetwork.backends.numpy import numpy_backend
from tensornetwork.backends.jax import jax_backend
#pylint: disable=no-member
config.update("jax_enable_x64", True)

np_real_dtypes = [np.float32, np.float16, np.float64]
np_dtypes = np_real_dtypes + [np.complex64, np.complex128]
np_int_dtypes = [np.int8, np.int16, np.int32, np.int64]
np_uint_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]

def test_eye_np():
  """
  Tests linalg.eye against np.eye.
  """
  N = 4
  M = 6
  name = "Jeffrey"
  axis_names = ["Sam", "Blinkey"]
  for dtype in np_dtypes + np_int_dtypes + np_uint_dtypes:
    tnI = linalg.eye(N, dtype=dtype, M=M, name=name, axis_names=axis_names,
                     backend=numpy_backend.NumPyBackend())
    npI = np.eye(N, M=M, dtype=dtype)
    np.testing.assert_allclose(tnI.tensor, npI)


def test_eye_jax():
  """
  Tests linalg.eye against jnp.eye.
  """
  N = 4
  M = 6
  name = "Jeffrey"
  axis_names = ["Sam", "Blinkey"]
  for dtype in np_dtypes + np_int_dtypes + np_uint_dtypes:
    tnI = linalg.eye(N, dtype=dtype, M=M, name=name, axis_names=axis_names,
                     backend=jax_backend.JaxBackend())
    jnpI = jnp.eye(N, M=M, dtype=dtype)
    np.testing.assert_allclose(tnI.tensor, jnpI)
