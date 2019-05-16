"""Tests for graphmode_tensornetwork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tf.enable_v2_behavior()
import tensornetwork


class GraphmodeTensorNetworkTest(tf.test.TestCase):

  def test_basic_graphmode(self):
    with tf.compat.v1.Graph().as_default():
      net = tensornetwork.TensorNetwork()
      a = net.add_node(tf.ones(10))
      b = net.add_node(tf.ones(10))
      e = net.connect(a[0], b[0])
      final_tensor = net.contract(e).get_tensor()

      sess = tf.compat.v1.Session()
      final_val = sess.run(final_tensor)
      self.assertAllClose(final_val, 10.0)

  def test_gradient_decent(self):
    with tf.compat.v1.Graph().as_default():
      net = tensornetwork.TensorNetwork()
      a = net.add_node(tf.Variable(tf.ones(10)))
      b = net.add_node(tf.ones(10))
      e = net.connect(a[0], b[0])
      final_tensor = net.contract(e).get_tensor()
      opt = tf.train.GradientDescentOptimizer(0.001)
      train_op = opt.minimize(final_tensor)
      sess = tf.compat.v1.Session()
      sess.run(tf.global_variables_initializer())
      self.assertAllClose(sess.run(final_tensor), 10.0)
      sess.run(train_op)
      self.assertLess(sess.run(final_tensor), 10.0)

if __name__ == '__main__':
  tf.test.main()
