# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
JAX_PRECISION = ["DEFAULT"]


def get_jax_precision(jax):
  if JAX_PRECISION[0] == "DEFAULT":
    return jax.lax.Precision.DEFAULT
  if JAX_PRECISION[0] == "HIGH":
    return jax.lax.Precision.HIGH
  if JAX_PRECISION[0] == "HIGHEST":
    return jax.lax.Precision.HIGHEST
  raise ValueError(f"found unknown value JAX_PRECISOIN={JAX_PRECISION}.")

def set_jax_precision(value):
  if value not in ("DEFAULT", "HIGH", "HIGHEST"):
    raise ValueError(f'{value} is not a valid value'
                     f' for JAX_PRECISION. Use "DEFAULT", "HIGH" '
                     f'or "HIGHEST"')
  JAX_PRECISION[0] = value
