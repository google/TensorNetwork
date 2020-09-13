import pytest
from tensornetwork.backends.jax.precision import get_jax_precision
import jax


def test_get_jax_precision():
  assert get_jax_precision(jax, "DEFAULT") is jax.lax.Precision.DEFAULT
  assert get_jax_precision(jax, "HIGH") is jax.lax.Precision.HIGH
  assert get_jax_precision(jax, "HIGHEST") is jax.lax.Precision.HIGHEST
  with pytest.raises(ValueError, match="unkown value NO_PRECISION"):
    get_jax_precision(jax, "NO_PRECISION")
