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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensornetwork
import pytest
import numpy as np

import tensorflow as tf

from jax.config import config

config.update("jax_enable_x64", True)
tf.compat.v1.enable_v2_behavior()


def test_normalization(backend):

  D, d, N = 10, 2, 10
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  mps = tensornetwork.mps.FiniteMPS(tensors, center_position=0, backend=backend)
  mps.position(len(mps) - 1)
  Z = mps.position(0, normalize=True)
  np.testing.assert_allclose(Z, 1.0)


@pytest.mark.parametrize("N, pos", [(10, -1), (10, 10)])
def test_mps_init(backend, N, pos):
  D, d, N = 10, 2, N
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  with pytest.raises(ValueError):
    mps = tensornetwork.mps.FiniteMPS(
        tensors, center_position=pos, backend=backend)


def test_left_orthonormalization(backend):
  D, d, N = 10, 2, 10
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  mps = tensornetwork.mps.FiniteMPS(
      tensors, center_position=N - 1, backend=backend)
  mps.position(0)
  mps.position(len(mps) - 1)
  assert all([
      mps.check_orthonormality('left', site) < 1E-12
      for site in range(len(mps))
  ])


def test_right_orthonormalization(backend):
  D, d, N = 10, 2, 10
  tensors = [np.random.randn(1, d, D)] + [
      np.random.randn(D, d, D) for _ in range(N - 2)
  ] + [np.random.randn(D, d, 1)]
  mps = tensornetwork.mps.FiniteMPS(tensors, center_position=0, backend=backend)

  mps.position(len(mps) - 1)
  mps.position(0)
  assert all([
      mps.check_orthonormality('right', site) < 1E-12
      for site in range(len(mps))
  ])
