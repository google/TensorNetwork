"""Tests for graphmode_tensornetwork."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensornetwork
from unittest import TestCase
from tensornetwork import set_default_backend
import tensorflow as tf


class BasicTensorFlowBackendTest(TestCase):

  @classmethod
  def setUpClass(cls):
    set_default_backend('tensorflow')

  def test_tensordot(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(2*tf.ones((2, 3, 4)))
    b = net.add_node(tf.ones((2, 3, 4)))
    tensordotted_tensor = net.backend.tensordot(a.tensor, b.tensor, ((1, 2), (1, 2)))
    self.assertListEqual(np.array(tensordotted_tensor).tolist(), [[24.0, 24.0], [24.0, 24.0]])

  def test_reshape(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones((2, 3, 4)))
    reshaped_tensor = net.backend.reshape(a.tensor, np.array((6, 4, 1)))
    self.assertTupleEqual(net.backend.shape_tuple(reshaped_tensor), (6, 4, 1))

  def test_transpose(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]))
    transposed_tensor = net.backend.transpose(a.tensor, [2, 0, 1])
    self.assertListEqual(np.array(transposed_tensor).tolist(), [[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]])

  def test_concat(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(2*tf.ones((1, 3, 1)))
    b = net.add_node(tf.ones((1, 2, 1)))
    concatenated_tensor = net.backend.concat((a.tensor, b.tensor), axis=1)
    self.assertListEqual(np.array(concatenated_tensor).tolist(), [[[2.0], [2.0], [2.0], [1.0], [1.0]]])

  def test_shape(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(tf.ones([2, 3, 4]))
    self.assertEqual(type(net.backend.shape(a.tensor)), type(a.tensor))
    self.assertListEqual(np.array(net.backend.shape(a.tensor)).tolist(), [2, 3, 4])

  def test_shape_tuple(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(tf.ones([2, 3, 4]))
    self.assertTupleEqual(net.backend.shape_tuple(a.tensor), (2, 3, 4))

  def test_prod(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(2*np.ones([1, 2, 3, 4]))
    self.assertEqual(np.array(net.backend.prod(a.tensor)), 2**24)

  def test_sqrt(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.array([4., 9.]))
    sqrt = net.backend.sqrt(a.tensor)
    self.assertListEqual(np.array(sqrt).tolist(), [2, 3])

  def test_diag(self):
    net = tensornetwork.TensorNetwork()
    b = net.add_node(np.array([1, 2, 3]))
    diagonal = net.backend.diag(b.tensor)
    self.assertListEqual(np.array(diagonal).tolist(), [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

  def test_convert_to_tensor(self):
    net = tensornetwork.TensorNetwork()
    array = np.ones((2, 3, 4))
    a = net.add_node(array)
    converted_to_tensor = net.backend.convert_to_tensor(array)
    self.assertEqual(type(converted_to_tensor), type(a.tensor))
    self.assertListEqual(np.array(converted_to_tensor).tolist(), np.ones((2, 3, 4)).tolist())

  def test_trace(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.array([[1, 2, 3], [4, 5, 6]]))
    trace = net.backend.trace(a.tensor)
    self.assertEqual(np.array(trace), 6)

  def test_outer_product(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(2 * np.ones((2, 1)))
    b = net.add_node(np.ones((1, 2, 2)))
    outer_product_tensor = net.backend.outer_product(a.tensor, b.tensor)
    self.assertListEqual(np.array(outer_product_tensor).tolist(),
                         [[[[[2.0, 2.0], [2.0, 2.0]]]], [[[[2.0, 2.0], [2.0, 2.0]]]]])

  def test_einsum(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(2 * np.ones((2, 1)))
    b = net.add_node(np.ones((1, 2, 2)))
    einsummed_tensor = net.backend.einsum('ij,jil->l', a.tensor, b.tensor)
    self.assertListEqual(np.array(einsummed_tensor).tolist(), [4.0, 4.0])


if __name__ == '__main__':
  tf.test.main()
