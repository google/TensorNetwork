import pytest
from tensornetwork.backends.jax.precision import (get_jax_precision,
                                                  set_jax_precision)
import jax

def test_set_jax_precision():
  set_jax_precision("DEFAULT")
  assert get_jax_precision(jax) is jax.lax.Precision.DEFAULT
  set_jax_precision("HIGH")
  assert get_jax_precision(jax) is jax.lax.Precision.HIGH
  set_jax_precision("HIGHEST")
  assert get_jax_precision(jax) is jax.lax.Precision.HIGHEST
  with pytest.raises(ValueError, match="WRONG_VALUE is not a valid"):
    set_jax_precision("WRONG_VALUE")
