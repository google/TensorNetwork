"""Tests for graphmode_tensornetwork."""
import numpy as np
import tensorflow as tf
from tensornetwork import (contract, connect, flatten_edges_between,
                           contract_between, Node)
import pytest


class GraphmodeTensorNetworkTest(tf.test.TestCase):

  def test_basic_graphmode(self):
    # pylint: disable=not-context-manager
    with tf.compat.v1.Graph().as_default():
      a = Node(tf.ones(10), backend="tensorflow")
      b = Node(tf.ones(10), backend="tensorflow")
      e = connect(a[0], b[0])
      final_tensor = contract(e).get_tensor()

      sess = tf.compat.v1.Session()
      final_val = sess.run(final_tensor)
      self.assertAllClose(final_val, 10.0)

  def test_gradient_decent(self):
    # pylint: disable=not-context-manager
    with tf.compat.v1.Graph().as_default():
      a = Node(tf.Variable(tf.ones(10)), backend="tensorflow")
      b = Node(tf.ones(10), backend="tensorflow")
      e = connect(a[0], b[0])
      final_tensor = contract(e).get_tensor()
      opt = tf.compat.v1.train.GradientDescentOptimizer(0.001)
      train_op = opt.minimize(final_tensor)
      sess = tf.compat.v1.Session()
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertAllClose(sess.run(final_tensor), 10.0)
      sess.run(train_op)
      self.assertLess(sess.run(final_tensor), 10.0)

  def test_dynamic_network_sizes(self):

    @tf.function
    def f(x, n):
      x_slice = x[:n]
      n1 = Node(x_slice, backend="tensorflow")
      n2 = Node(x_slice, backend="tensorflow")
      e = connect(n1[0], n2[0])
      return contract(e).get_tensor()

    x = np.ones(10)
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), 2.0)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), 3.0)

  @pytest.mark.skip(reason="Test fails due to probable bug in tensorflow 2.0.0")
  def test_dynamic_network_sizes_contract_between(self):

    @tf.function
    def f(x, n):
      x_slice = x[..., :n]
      n1 = Node(x_slice, backend="tensorflow")
      n2 = Node(x_slice, backend="tensorflow")
      connect(n1[0], n2[0])
      connect(n1[1], n2[1])
      connect(n1[2], n2[2])
      return contract_between(n1, n2).get_tensor()

    x = tf.ones((3, 4, 5))
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), 24.0)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), 36.0)

  def test_dynamic_network_sizes_flatten_standard(self):

    @tf.function
    def f(x, n):
      x_slice = x[..., :n]
      n1 = Node(x_slice, backend="tensorflow")
      n2 = Node(x_slice, backend="tensorflow")
      connect(n1[0], n2[0])
      connect(n1[1], n2[1])
      connect(n1[2], n2[2])
      return contract(flatten_edges_between(n1, n2)).get_tensor()

    x = np.ones((3, 4, 5))
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), 24.0)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), 36.0)

  def test_dynamic_network_sizes_flatten_trace(self):

    @tf.function
    def f(x, n):
      x_slice = x[..., :n]
      n1 = Node(x_slice, backend="tensorflow")
      connect(n1[0], n1[2])
      connect(n1[1], n1[3])
      return contract(flatten_edges_between(n1, n1)).get_tensor()

    x = np.ones((3, 4, 3, 4, 5))
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), np.ones((2,)) * 12)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), np.ones((3,)) * 12)

  def test_batch_usage(self,):

    def build_tensornetwork(tensors):
      a = Node(tensors[0], backend="tensorflow")
      b = Node(tensors[1], backend="tensorflow")
      e = connect(a[0], b[0])
      return contract(e).get_tensor()

    tensors = [np.ones((5, 10)), np.ones((5, 10))]
    result = tf.map_fn(build_tensornetwork, tensors, dtype=tf.float64)
    np.testing.assert_allclose(result, np.ones(5) * 10)


if __name__ == '__main__':
  tf.test.main()
