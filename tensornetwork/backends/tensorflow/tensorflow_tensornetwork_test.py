"""Tests for graphmode_tensornetwork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
import tensornetwork


class GraphmodeTensorNetworkTest(tf.test.TestCase):

  def test_basic_graphmode(self):
    # pylint: disable=not-context-manager
    with tf.compat.v1.Graph().as_default():
      net = tensornetwork.TensorNetwork(backend="tensorflow")
      a = net.add_node(tf.ones(10))
      b = net.add_node(tf.ones(10))
      e = net.connect(a[0], b[0])
      final_tensor = net.contract(e).get_tensor()

      sess = tf.compat.v1.Session()
      final_val = sess.run(final_tensor)
      self.assertAllClose(final_val, 10.0)

  def test_gradient_decent(self):
    # pylint: disable=not-context-manager
    with tf.compat.v1.Graph().as_default():
      net = tensornetwork.TensorNetwork(backend="tensorflow")
      a = net.add_node(tf.Variable(tf.ones(10)))
      b = net.add_node(tf.ones(10))
      e = net.connect(a[0], b[0])
      final_tensor = net.contract(e).get_tensor()
      opt = tf.compat.v1.train.GradientDescentOptimizer(0.001)
      train_op = opt.minimize(final_tensor)
      sess = tf.compat.v1.Session()
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertAllClose(sess.run(final_tensor), 10.0)
      sess.run(train_op)
      self.assertLess(sess.run(final_tensor), 10.0)

  def test_dynamic_network_sizes(self):

    @tf.contrib.eager.defun
    def f(x, n):
      x_slice = x[:n]
      net = tensornetwork.TensorNetwork(backend="tensorflow")
      n1 = net.add_node(x_slice)
      n2 = net.add_node(x_slice)
      e = net.connect(n1[0], n2[0])
      return net.contract(e).get_tensor()

    x = np.ones(10)
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), 2.0)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), 3.0)

  def test_dynamic_network_sizes_contract_between(self):

    @tf.contrib.eager.defun
    def f(x, n):
      x_slice = x[..., :n]
      net = tensornetwork.TensorNetwork(backend="tensorflow")
      n1 = net.add_node(x_slice)
      n2 = net.add_node(x_slice)
      net.connect(n1[0], n2[0])
      net.connect(n1[1], n2[1])
      net.connect(n1[2], n2[2])
      return net.contract_between(n1, n2).get_tensor()

    x = tf.ones((3, 4, 5))
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), 24.0)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), 36.0)

  def test_dynamic_network_sizes_flatten_standard(self):

    @tf.contrib.eager.defun
    def f(x, n):
      x_slice = x[..., :n]
      net = tensornetwork.TensorNetwork(backend="tensorflow")
      n1 = net.add_node(x_slice)
      n2 = net.add_node(x_slice)
      net.connect(n1[0], n2[0])
      net.connect(n1[1], n2[1])
      net.connect(n1[2], n2[2])
      return net.contract(net.flatten_edges_between(n1, n2)).get_tensor()

    x = np.ones((3, 4, 5))
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), 24.0)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), 36.0)

  def test_dynamic_network_sizes_flatten_trace(self):

    @tf.contrib.eager.defun
    def f(x, n):
      x_slice = x[..., :n]
      net = tensornetwork.TensorNetwork(backend="tensorflow")
      n1 = net.add_node(x_slice)
      net.connect(n1[0], n1[2])
      net.connect(n1[1], n1[3])
      return net.contract(net.flatten_edges_between(n1, n1)).get_tensor()

    x = np.ones((3, 4, 3, 4, 5))
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), np.ones((2,)) * 12)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), np.ones((3,)) * 12)

  def test_batch_usage(self,):

    def build_tensornetwork(tensors):
      net = tensornetwork.TensorNetwork(backend="tensorflow")
      a = net.add_node(tensors[0])
      b = net.add_node(tensors[1])
      e = net.connect(a[0], b[0])
      return net.contract(e).get_tensor()

    tensors = [np.ones((5, 10)), np.ones((5, 10))]
    result = tf.map_fn(build_tensornetwork, tensors, dtype=tf.float64)
    np.testing.assert_allclose(result, np.ones(5) * 10)

if __name__ == '__main__':
  tf.test.main()
