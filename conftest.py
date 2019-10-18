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

import pytest
import jax
import tensornetwork
import tensorflow as tf


@pytest.fixture(
    name="backend", params=["numpy", "tensorflow", "jax", "pytorch"])
def backend_fixture(request):
  return request.param


@pytest.fixture(autouse=True)
def reset_default_backend():
  tensornetwork.set_default_backend("numpy")
  yield
  tensornetwork.set_default_backend("numpy")


@pytest.fixture(autouse=True)
def enable_jax_64():
  jax.config.update("jax_enable_x64", True)
  yield
  jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def tf_enable_v2_behaviour():
  tf.compat.v1.enable_v2_behavior()
  yield
  tf.compat.v1.enable_v2_behavior()
