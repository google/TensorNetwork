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

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tensornetwork.backends.tensorflow.tensordot2"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensornetwork.backends.tensorflow import tensordot2
import pytest

tf.compat.v1.enable_v2_behavior()
_MAXDIM = 5


class TensordotTest(tf.compat.v1.test.TestCase):

  def test_invalid_shape(self):
    a = [[1, 2], [3, 4]]
    b = [[1, 2], [3, 4], [5, 6]]
    a_axes = [1]
    b_axes = [0]
    # Invalid static shapes.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      tensordot2.tensordot(a, b, (a_axes, b_axes))
    # Invalid dynamic shapes.
    # pylint: disable=not-context-manager
    with tf.compat.v1.Graph().as_default():
      with self.cached_session() as sess:
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                     "Matrix size-incompatible"):
          a_ph = tf.compat.v1.placeholder(tf.float32)
          b_ph = tf.compat.v1.placeholder(tf.float32)
          axes_ph = tf.compat.v1.placeholder(tf.int32)
          output = tensordot2.tensordot(a_ph, b_ph, axes_ph)
          _ = sess.run(
              [output], feed_dict={
                  a_ph: a,
                  b_ph: b,
                  axes_ph: (a_axes, b_axes)
              })

  def test_invalid_axes(self):
    # pylint: disable=not-context-manager
    with tf.compat.v1.Graph().as_default():
      a = [[1, 2], [3, 4]]
      b = [[1, 2], [3, 4]]
      # Invalid static axes.
      for axes_value in -1, 3, [1], [[1]], [[1], [0, 1]]:
        with self.assertRaises(ValueError):
          tensordot2.tensordot(a, b, axes_value)

      with self.assertRaises(IndexError):
        tensordot2.tensordot(a, b, [[0], [7]])

      # Invalid dynamic axes.
      a_ph = tf.compat.v1.placeholder(tf.float32)
      b_ph = tf.compat.v1.placeholder(tf.float32)
      axes_ph = tf.compat.v1.placeholder(tf.int32)
      output = tensordot2.tensordot(a_ph, b_ph, axes_ph)
      # Note: We don't support scalar Tensor values for axes.
      for axes_value in 1, [1], [0, 1], [[1]], [[0, 1]], [[0], [7]]:
        with self.cached_session() as sess:
          with self.assertRaises(tf.errors.InvalidArgumentError):
            _ = sess.run(
                [output], feed_dict={
                    a_ph: a,
                    b_ph: b,
                    axes_ph: axes_value
                })

  # Test case for 11950
  def test_valid_axis(self):
    for axes_value in [1, 2], [[1], [2]], [[], []], 0:
      with self.cached_session():
        np_a = np.ones((3, 3))
        np_b = np.array([2, 3, 1])[None, None]
        np_ans = np.tensordot(np_a, np_b, axes_value)

        tf_a = tf.ones((3, 3), dtype=tf.float32)
        tf_b = tf.constant([2, 3, 1], dtype=tf.float32)[None, None]
        tf_ans = tensordot2.tensordot(tf_a, tf_b, axes_value)

        self.assertAllEqual(tf_ans.shape, np_ans.shape)
        self.assertAllEqual(tf_ans, np_ans)

  def test_partial_shape_inference(self):
    # pylint: disable=not-context-manager
    with tf.compat.v1.Graph().as_default():
      for axes in ([1], [0]), 1:
        a = tf.compat.v1.placeholder(tf.float32)
        b = tf.compat.v1.placeholder(tf.float32)
        output = tensordot2.tensordot(a, b, axes)
        self.assertEqual(output.get_shape().ndims, None)
        a.set_shape([None, 2])
        b.set_shape([2, 3])
        output = tensordot2.tensordot(a, b, axes)
        output_shape = output.get_shape()
        self.assertEqual(output_shape.ndims, 2)
        output_shape = output_shape.as_list()
        self.assertEqual(output_shape[0], None)
        self.assertEqual(output_shape[1], 3)
        a = tf.compat.v1.placeholder(tf.float32)
        b = tf.compat.v1.placeholder(tf.float32)
        a.set_shape([2, 2])
        b.set_shape([2, None])
        output = tensordot2.tensordot(a, b, axes)
        output_shape = output.get_shape()
        self.assertEqual(output_shape.ndims, 2)
        output_shape = output_shape.as_list()
        self.assertEqual(output_shape[0], 2)
        self.assertEqual(output_shape[1], None)


# Select a random subset of size m from [0, 1, ..., n-1].
def _random_subset(m, n):
  assert m <= n
  return (np.random.permutation(n)[:m]).astype(np.int32)

def _generate_random_tensors_and_dims(dtype_, rank_a_, rank_b_, num_dims_):
  a_shape = np.random.randint(1, _MAXDIM + 1, rank_a_)
  b_shape = np.random.randint(1, _MAXDIM + 1, rank_b_)
  shared_shape = np.random.randint(1, _MAXDIM + 1, num_dims_)
  a_dims = _random_subset(num_dims_, rank_a_)
  b_dims = _random_subset(num_dims_, rank_b_)
  for i in range(num_dims_):
    a_shape[a_dims[i]] = shared_shape[i]
    b_shape[b_dims[i]] = shared_shape[i]
  a = np.random.uniform(
      low=-1.0, high=1.0,
      size=np.prod(a_shape)).reshape(a_shape).astype(dtype_)
  b = np.random.uniform(
      low=-1.0, high=1.0,
      size=np.prod(b_shape)).reshape(b_shape).astype(dtype_)
  return a, b, a_dims, b_dims

@pytest.mark.parametrize("dtype_", [np.float32, np.complex64])
@pytest.mark.parametrize("rank_a_", [1, 2, 3])
@pytest.mark.parametrize("rank_b_", [1, 2, 3])
@pytest.mark.parametrize("num_dims_", [1, 2, 3])
def test_tensordot_scalar_axes(dtype_, rank_a_, rank_b_, num_dims_):
  if not (num_dims_ <= min(rank_a_, rank_b_)):
    pytest.skip("Not a test")
  if dtype_ == np.float16:
    tol = 0.05
  elif dtype_ in (np.float32, np.complex64):
    tol = 1e-5
  else:
    tol = 1e-12
  shape = [5] * num_dims_
  a_np = np.random.uniform(
      low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype_)
  b_np = np.random.uniform(
      low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype_)
  all_axes = [0, 1]
  if a_np.ndim > 2:
    all_axes.append(a_np.ndim - 1)
  for axes in all_axes:
    np_ans = np.tensordot(a_np, b_np, axes=axes)
    tf_ans = tensordot2.tensordot(a_np, b_np, axes=axes)
    np.testing.assert_allclose(tf_ans, np_ans, rtol=tol, atol=tol)
    assert tf_ans.shape == np_ans.shape

@pytest.mark.parametrize("dtype_", [np.float32, np.complex64])
@pytest.mark.parametrize("rank_a_", [1, 2, 3])
@pytest.mark.parametrize("rank_b_", [1, 2, 3])
@pytest.mark.parametrize("num_dims_", [0, 1, 2, 3])
def test_tensordot(dtype_, rank_a_, rank_b_, num_dims_):
  if not (num_dims_ <= min(rank_a_, rank_b_)):
    pytest.skip("Not a test")
  num_trials = min(30, num_dims_ * num_dims_)
  if dtype_ == np.float16:
    tol = 0.05
  elif dtype_ in (np.float32, np.complex64):
    tol = 1e-5
  else:
    tol = 1e-12
  for _ in range(num_trials):
    a_np, b_np, a_dims_np, b_dims_np = _generate_random_tensors_and_dims(
        dtype_, rank_a_, rank_b_, num_dims_)
    np_ans = np.tensordot(a_np, b_np, axes=(a_dims_np, b_dims_np))
    tf_ans = tensordot2.tensordot(a_np, b_np, (a_dims_np, b_dims_np))
    np.testing.assert_allclose(tf_ans, np_ans, rtol=tol, atol=tol)
    assert tf_ans.shape == np_ans.shape
