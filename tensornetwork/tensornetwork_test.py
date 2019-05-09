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
import numpy as np
import tensorflow as tf
# Prepare for TF 2.0 migration
tf.enable_v2_behavior()
# pylint: disable=g-import-not-at-top
from tensornetwork import tensornetwork


class NetworkTest(tf.test.TestCase):

  def test_sanity_check(self):
    net = tensornetwork.TensorNetwork()
    net.add_node(np.eye(2), "a")
    net.check_correct()

  def test_node_names(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.eye(2), "a", axis_names=["e0", "e1"])
    self.assertEqual(a.name, "a")
    self.assertEqual(a[0].name, "e0")
    self.assertEqual(a[1].name, "e1")

  def test_single_contract(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.array([1.0] * 5), "a")
    b = net.add_node(np.array([1.0] * 5), "b")
    e = net.connect(a[0], b[0])
    c = net.contract(e)
    net.check_correct()
    val = c.get_tensor().numpy()
    self.assertAlmostEqual(val, 5.0)

  def test_disconnect_edge(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.array([1.0] * 5), "a")
    b = net.add_node(np.array([1.0] * 5), "b")
    e = net.connect(a[0], b[0])
    self.assertFalse(e.is_dangling())
    dangling_edge_1, dangling_edge_2 = net.disconnect(e)
    net.check_correct(check_connected=False)
    self.assertTrue(dangling_edge_1.is_dangling())
    self.assertTrue(dangling_edge_2.is_dangling())
    self.assertEqual(a.get_edge(0), dangling_edge_1)
    self.assertEqual(b.get_edge(0), dangling_edge_2)

  def test_set_tensor(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones(2))
    self.assertAllClose(a.get_tensor(), np.ones(2))
    a.set_tensor(np.zeros(2))
    self.assertAllClose(a.get_tensor(), np.zeros(2))

  def test_has_nondangling_edge(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones(2))
    self.assertFalse(a.has_nondangling_edge())
    b = net.add_node(np.ones((2, 2)))
    net.connect(b[0], b[1])
    self.assertTrue(b.has_nondangling_edge())

  def test_large_nodes(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros([5, 6, 7, 8, 9]), "a")
    b = net.add_node(np.zeros([5, 6, 7, 8, 9]), "b")
    for i in range(5):
      net.connect(a[i], b[i])
    net.check_correct()

  def test_small_matmul(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros([10, 10]), name="a")
    b = net.add_node(np.zeros([10, 10]), name="b")
    edge = net.connect(a[0], b[0], "edge")
    net.check_correct()
    c = net.contract(edge, name="a * b")
    self.assertEqual(c.get_tensor().shape, [10, 10])
    net.check_correct()

  def test_direct_trace(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones([10, 10]), name="a")
    edge = net.connect(a[0], a[1], "edge")
    net.check_correct()
    result = net._contract_trace(edge)
    net.check_correct()
    self.assertAlmostEqual(result.get_tensor().numpy(), 10.0)

  def test_double_trace(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones([10, 10, 10, 10]), name="a")
    edge1 = net.connect(a[0], a[1], "edge1")
    edge2 = net.connect(a[2], a[3], "edge2")
    net.check_correct()
    net._contract_trace(edge1)
    net.check_correct()
    val = net._contract_trace(edge2)
    net.check_correct()
    self.assertAlmostEqual(val.get_tensor().numpy(), 100.0)

  def test_indirect_trace(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones([10, 10]), name="a")
    edge = net.connect(a[0], a[1], "edge")
    net.check_correct()
    val = net.contract(edge)
    net.check_correct()
    self.assertAlmostEqual(val.get_tensor().numpy(), 10.0)

  def test_real_physics(self):
    # Calcuate the expected value in numpy
    a_vals = np.ones([2, 3, 4, 5])
    b_vals = np.ones([4, 6, 7])
    c_vals = np.ones([5, 6, 8])
    contract1 = np.tensordot(a_vals, b_vals, [[2], [0]])
    contract2 = np.tensordot(c_vals, contract1, [[0], [2]])
    final_result = np.trace(contract2, axis1=0, axis2=4)
    # Build the network
    net = tensornetwork.TensorNetwork()
    a = net.add_node(a_vals, name="T")
    b = net.add_node(b_vals, name="A")
    c = net.add_node(c_vals, name="B")
    e1 = net.connect(a[2], b[0], "edge")
    e2 = net.connect(c[0], a[3], "edge2")
    e3 = net.connect(b[1], c[1], "edge3")
    net.check_correct()
    node_result = net.contract(e1)
    self.assertAllClose(node_result.get_tensor(), contract1)
    net.check_correct()
    node_result = net.contract(e2)
    self.assertAllClose(node_result.get_tensor(), contract2)
    net.check_correct()
    val = net.contract(e3)
    net.check_correct()
    self.assertAllClose(val.get_tensor(), final_result)

  def test_real_physics_with_tensors(self):
    # Calcuate the expected value in numpy
    a_vals = np.ones([2, 3, 4, 5])
    b_vals = np.ones([4, 6, 7])
    c_vals = np.ones([5, 6, 8])
    contract1 = np.tensordot(a_vals, b_vals, [[2], [0]])
    contract2 = np.tensordot(c_vals, contract1, [[0], [2]])
    final_result = np.trace(contract2, axis1=0, axis2=4)
    # Build the network
    net = tensornetwork.TensorNetwork()
    a = net.add_node(tf.ones([2, 3, 4, 5]), name="T")
    b = net.add_node(tf.ones([4, 6, 7]), name="A")
    c = net.add_node(tf.ones([5, 6, 8]), name="B")
    e1 = net.connect(a[2], b[0], "edge")
    e2 = net.connect(c[0], a[3], "edge2")
    e3 = net.connect(b[1], c[1], "edge3")
    net.check_correct()
    node_result = net.contract(e1)
    self.assertAllClose(node_result.get_tensor(), contract1)
    net.check_correct()
    node_result = net.contract(e2)
    self.assertAllClose(node_result.get_tensor(), contract2)
    net.check_correct()
    val = net.contract(e3)
    net.check_correct()
    self.assertAllClose(val.get_tensor(), final_result)

  def test_real_physics_naive_contraction(self):
    # Calcuate the expected value in numpy
    a_vals = np.ones([2, 3, 4, 5])
    b_vals = np.ones([4, 6, 7])
    c_vals = np.ones([5, 6, 8])
    contract1 = np.tensordot(a_vals, b_vals, [[2], [0]])
    contract2 = np.tensordot(c_vals, contract1, [[0], [2]])
    final_result = np.trace(contract2, axis1=0, axis2=4)
    # Build the network
    net = tensornetwork.TensorNetwork()
    a = net.add_node(tf.ones([2, 3, 4, 5]), name="T")
    b = net.add_node(tf.ones([4, 6, 7]), name="A")
    c = net.add_node(tf.ones([5, 6, 8]), name="B")
    e1 = net.connect(a[2], b[0], "edge")
    e2 = net.connect(c[0], a[3], "edge2")
    e3 = net.connect(b[1], c[1], "edge3")
    for edge in [e1, e2, e3]:
      net.contract(edge)
    val = net.get_final_node()
    self.assertEqual(val.get_tensor().shape, [8, 2, 3, 7])
    self.assertAllClose(val.get_tensor(), final_result)

  def test_with_tensors(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(tf.eye(2) * 2, name="T")
    b = net.add_node(tf.eye(2) * 3, name="A")
    e1 = net.connect(a[0], b[0], "edge")
    e2 = net.connect(a[1], b[1], "edge2")
    net.check_correct()
    net.contract(e1)
    net.check_correct()
    val = net.contract(e2)
    net.check_correct()
    self.assertAlmostEqual(val.get_tensor().numpy(), 12.0)

  def test_contract_dangling_edge(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.array([1]))
    e = a[0]
    with self.assertRaises(ValueError):
      net.contract(e)

  def test_double_edge_contract(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.eye(2))
    e = net.connect(a[0], a[1], name="edge")
    net.contract(e)
    with self.assertRaises(ValueError):
      net.contract(e)

  def test_contract_trace_dangling_edge(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.array([1]))
    e = a[0]
    with self.assertRaises(ValueError):
      net._contract_trace(e)

  def test_node2_contract_trace(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros([3, 3, 1]))
    b = net.add_node(np.zeros([1]))
    net.connect(b[0], a[2])
    trace_edge = net.connect(a[0], a[1])
    net._contract_trace(trace_edge)
    net.check_correct()

  def test_contract_fall_through_name(self):
    net = tensornetwork.TensorNetwork()
    node = net.add_node(np.eye(2), name="Identity Matrix")
    self.assertEqual(node.name, "Identity Matrix")
    edge = net.connect(node[0], node[1], name="Trace Edge")
    self.assertEqual(edge.name, "Trace Edge")
    final_result = net.contract(edge, name="Trace Of Identity")
    self.assertEqual(final_result.name, "Trace Of Identity")

  def test_non_connected(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.array([2, 2]))
    b = net.add_node(np.array([2, 2]))
    net.connect(a[0], b[0])
    c = net.add_node(np.array([2, 2]))
    d = net.add_node(np.array([2, 2]))
    net.connect(c[0], d[0])
    with self.assertRaises(ValueError):
      net.check_connected()

  def test_node_get_dim_bad_axis(self):
    node = tensornetwork.Node(np.eye(2), "a", axis_names=["1", "2"])
    with self.assertRaises(ValueError):
      node.get_dimension(10)

  def test_bad_trace_contract(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.array([2]))
    b = net.add_node(np.array([2]))
    e = net.connect(a[0], b[0])
    with self.assertRaises(ValueError):
      net._contract_trace(e)

  def test_double_edge_axis(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.array([2]), name="a")
    b = net.add_node(np.array([2]), name="b")
    net.connect(a[0], b[0])
    with self.assertRaises(ValueError):
      net.connect(a[0], b[0])

  def test_named_axis(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.eye(2), axis_names=["alpha", "beta"])
    e = net.connect(a["alpha"], a["beta"])
    b = net.contract(e)
    self.assertAlmostEqual(b.get_tensor().numpy(), 2.0)

  def test_mixed_named_axis(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.eye(2) * 2.0, axis_names=["alpha", "beta"])
    b = net.add_node(np.eye(2) * 3.0)
    e1 = net.connect(a["alpha"], b[0])
    # Axes should still be indexable by numbers even with naming.
    e2 = net.connect(a[1], b[1])
    net.contract(e1)
    result = net.contract(e2)
    self.assertAlmostEqual(result.get_tensor().numpy(), 12.0)

  def test_duplicate_name(self):
    net = tensornetwork.TensorNetwork()
    with self.assertRaises(ValueError):
      net.add_node(np.eye(2), axis_names=["test", "test"])

  def test_bad_axis_name_length(self):
    net = tensornetwork.TensorNetwork()
    with self.assertRaises(ValueError):
      # This should have 2 names, not 1.
      net.add_node(np.eye(2), axis_names=["need_2_names"])

  def test_bad_axis_name_connect(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.eye(2), axis_names=["test", "names"])
    with self.assertRaises(ValueError):
      a.get_edge("bad_name")

  def test_node_edge_ordering(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros((2, 3, 4)))
    e2 = a[0]
    e3 = a[1]
    e4 = a[2]
    self.assertEqual(a.get_tensor().shape, (2, 3, 4))
    a.reorder_edges([e4, e2, e3])
    net.check_correct()
    self.assertEqual(a.get_tensor().shape, (4, 2, 3))
    self.assertEqual(e2.axis1, 1)
    self.assertEqual(e3.axis1, 2)
    self.assertEqual(e4.axis1, 0)

  def test_trace_edge_ordering(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros((2, 2, 3)))
    e2 = net.connect(a[1], a[0])
    e3 = a[2]
    with self.assertRaises(ValueError):
      a.reorder_edges([e2, e3])

  def test_mismatch_edge_ordering(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros((2, 3)))
    e2_a = a[0]
    b = net.add_node(np.zeros((2,)))
    e_b = b[0]
    with self.assertRaises(ValueError):
      a.reorder_edges([e2_a, e_b])

  def test_complicated_edge_reordering(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros((2, 3, 4)))
    b = net.add_node(np.zeros((2, 5)))
    c = net.add_node(np.zeros((3,)))
    d = net.add_node(np.zeros((4, 5)))
    e_ab = net.connect(a[0], b[0])
    e_bd = net.connect(b[1], d[1])
    e_ac = net.connect(a[1], c[0])
    e_ad = net.connect(a[2], d[0])
    net.contract(e_bd)
    a.reorder_edges([e_ac, e_ab, e_ad])
    net.check_correct()
    self.assertEqual(a.get_tensor().shape, (3, 2, 4))

  def test_edge_reorder_axis_names(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros((2, 3, 4, 5)), axis_names=["a", "b", "c", "d"])
    edge_a = a["a"]
    edge_b = a["b"]
    edge_c = a["c"]
    edge_d = a["d"]
    a.reorder_edges([edge_c, edge_b, edge_d, edge_a])
    self.assertEqual(a.get_tensor().shape, (4, 3, 5, 2))
    self.assertEqual(a.axis_names, ["c", "b", "d", "a"])

  def test_outer_product(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones((2, 4, 5)), name="A")
    b = net.add_node(np.ones((4, 3, 6)), name="B")
    c = net.add_node(np.ones((3, 2)), name="C")
    net.connect(a[1], b[0])
    net.connect(a[0], c[1])
    net.connect(b[1], c[0])
    # Purposely leave b's 3rd axis undefined.
    d = net.outer_product(a, b, name="D")
    net.check_correct()
    self.assertEqual(d.get_tensor().shape, (2, 4, 5, 4, 3, 6))
    self.assertAllClose(d.get_tensor().numpy(), np.ones((2, 4, 5, 4, 3, 6)))
    self.assertEqual(d.name, "D")

  def test_outer_product_final_nodes(self):
    net = tensornetwork.TensorNetwork()
    edges = []
    for i in range(1, 5):
      edges.append(net.add_node(tf.ones(i))[0])
    final_node = net.outer_product_final_nodes(edges)
    self.assertAllClose(final_node.get_tensor(), np.ones([1, 2, 3, 4]))
    self.assertEqual(final_node.get_all_edges(), edges)

  def test_outer_product_final_nodes_not_contracted(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones(2))
    b = net.add_node(np.ones(2))
    e = net.connect(a[0], b[0])
    with self.assertRaises(ValueError):
      net.outer_product_final_nodes([e])

  def test_add_axis_names(self):
    a = tensornetwork.Node(np.eye(2), "A", ["ignore1", "ignore2"])
    a.add_axis_names(["a", "b"])
    self.assertEqual(a.axis_names, ["a", "b"])

  def test_reorder_axes(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros((2, 3, 4)))
    b = net.add_node(np.zeros((3, 4, 5)))
    c = net.add_node(np.zeros((2, 4, 5)))
    net.connect(a[0], c[0])
    net.connect(b[0], a[1])
    net.connect(a[2], c[1])
    net.connect(b[2], c[2])
    a.reorder_axes([2, 0, 1])
    net.check_correct()
    self.assertEqual(a.get_tensor().shape, (4, 2, 3))

  def test_flattening_standard_edges(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros((2, 3, 5)), name="A")
    b = net.add_node(np.zeros((2, 3, 4, 5)), name="B")
    e1 = net.connect(a[0], b[0], "Edge_1_1")
    e2 = net.connect(a[2], b[3], "Edge_2_3")
    edge_a_1 = a[1]
    edge_b_1 = b[1]
    edge_b_2 = b[2]
    new_edge = net.flatten_edges([e1, e2], new_edge_name="New Edge")
    self.assertEqual(a.get_tensor().shape, (3, 10))
    self.assertEqual(b.get_tensor().shape, (3, 4, 10))
    self.assertEqual(a.edges, [edge_a_1, new_edge])
    self.assertEqual(b.edges, [edge_b_1, edge_b_2, new_edge])
    net.check_correct()

  def test_flattening_dangling_edges(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros((2, 3, 4, 5)), name="A")
    e1 = a[0]
    e2 = a[1]
    e3 = a[2]
    e4 = a[3]
    flattened_edge = net.flatten_edges([e1, e3], new_edge_name="New Edge")
    self.assertEqual(a.get_tensor().shape, (3, 5, 8))
    self.assertEqual(a.edges, [e2, e4, flattened_edge])
    self.assertEqual(flattened_edge.name, "New Edge")
    net.check_correct()

  def test_flatten_edges_different_nodes(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.eye(2))
    b = net.add_node(np.eye(2))
    c = net.add_node(np.eye(2))
    e1 = net.connect(a[0], b[0])
    e2 = net.connect(a[1], c[0])
    net.connect(b[1], c[1])
    with self.assertRaises(ValueError):
      net.flatten_edges([e1, e2])

  def test_flatten_trace_edges(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.zeros((2, 3, 4, 3, 5, 5)))
    c = net.add_node(np.zeros((2, 4)))
    e1 = net.connect(a[1], a[3])
    e2 = net.connect(a[4], a[5])
    external_1 = net.connect(a[0], c[0])
    external_2 = net.connect(c[1], a[2])
    new_edge = net.flatten_edges([e1, e2], "New Edge")
    net.check_correct()
    self.assertEqual(a.get_tensor().shape, (2, 4, 15, 15))
    self.assertEqual(a.edges, [external_1, external_2, new_edge, new_edge])
    self.assertEqual(new_edge.name, "New Edge")

  def test_flatten_consistent_result(self):
    net_noflat = tensornetwork.TensorNetwork()
    a_val = np.random.normal(size=(3, 5, 5, 6))
    b_val = np.random.normal(size=(5, 6, 4, 5))
    # Create non flattened example to compare against.
    a_noflat = net_noflat.add_node(a_val)
    b_noflat = net_noflat.add_node(b_val)
    e1 = net_noflat.connect(a_noflat[1], b_noflat[3])
    e2 = net_noflat.connect(a_noflat[3], b_noflat[1])
    e3 = net_noflat.connect(a_noflat[2], b_noflat[0])
    a_dangling_noflat = a_noflat[0]
    b_dangling_noflat = b_noflat[2]
    for edge in [e1, e2, e3]:
      net_noflat.contract(edge)
    noflat_result_node = net_noflat.get_final_node()
    noflat_result_node.reorder_edges([a_dangling_noflat, b_dangling_noflat])
    noflat_result = noflat_result_node.get_tensor().numpy()
    # Create network with flattening
    net_flat = tensornetwork.TensorNetwork()
    a_flat = net_flat.add_node(a_val)
    b_flat = net_flat.add_node(b_val)
    e1 = net_flat.connect(a_flat[1], b_flat[3])
    e2 = net_flat.connect(a_flat[3], b_flat[1])
    e3 = net_flat.connect(a_flat[2], b_flat[0])
    a_dangling_flat = a_flat[0]
    b_dangling_flat = b_flat[2]
    final_edge = net_flat.flatten_edges([e1, e2, e3])
    flat_result_node = net_flat.contract(final_edge)
    flat_result_node.reorder_edges([a_dangling_flat, b_dangling_flat])
    flat_result = flat_result_node.get_tensor().numpy()
    self.assertAllClose(flat_result, noflat_result)

  def test_flatten_consistent_tensor(self):
    net = tensornetwork.TensorNetwork()
    a_val = np.random.normal(size=(2, 3, 4, 5))
    b_val = np.random.normal(size=(3, 5, 4, 2))
    a = net.add_node(a_val)
    b = net.add_node(b_val)
    e1 = net.connect(a[0], b[3])
    e2 = net.connect(b[1], a[3])
    e3 = net.connect(a[1], b[0])
    net.flatten_edges([e3, e1, e2])
    net.check_correct()

    # Check expected values.
    a_final = np.reshape(np.transpose(a_val, (2, 1, 0, 3)), (4, 30))
    b_final = np.reshape(np.transpose(b_val, (2, 0, 3, 1)), (4, 30))
    self.assertAllClose(a.get_tensor().numpy(), a_final)
    self.assertAllClose(b.get_tensor().numpy(), b_final)

  def test_flatten_trace_consistent_result(self):
    net_noflat = tensornetwork.TensorNetwork()
    a_val = np.random.normal(size=(5, 6, 6, 7, 5, 7))
    a_noflat = net_noflat.add_node(a_val)
    e1 = net_noflat.connect(a_noflat[0], a_noflat[4])
    e2 = net_noflat.connect(a_noflat[1], a_noflat[2])
    e3 = net_noflat.connect(a_noflat[3], a_noflat[5])
    for edge in [e1, e2, e3]:
      net_noflat.contract(edge)
    noflat_result = net_noflat.get_final_node().get_tensor().numpy()
    # Create network with flattening
    net_flat = tensornetwork.TensorNetwork()
    a_flat = net_flat.add_node(a_val)
    e1 = net_flat.connect(a_flat[0], a_flat[4])
    e2 = net_flat.connect(a_flat[1], a_flat[2])
    e3 = net_flat.connect(a_flat[3], a_flat[5])
    final_edge = net_flat.flatten_edges([e1, e2, e3])
    flat_result = net_flat.contract(final_edge).get_tensor().numpy()
    self.assertAllClose(flat_result, noflat_result)

  def test_flatten_trace_consistent_tensor(self):
    net = tensornetwork.TensorNetwork()
    a_val = np.random.normal(size=(5, 3, 4, 4, 5))
    a = net.add_node(a_val)
    e1 = net.connect(a[0], a[4])
    e2 = net.connect(a[3], a[2])
    net.flatten_edges([e2, e1])
    net.check_correct()
    # Check expected values.
    a_final = np.reshape(np.transpose(a_val, (1, 2, 0, 3, 4)), (3, 20, 20))
    self.assertAllClose(a.get_tensor().numpy(), a_final)

  def test_add_subnetwork(self):
    net1 = tensornetwork.TensorNetwork()
    net2 = tensornetwork.TensorNetwork()
    a = net1.add_node(np.eye(2) * 2)
    b = net1.add_node(np.eye(2) * 3)
    e1 = net1.connect(a[0], b[0])
    c = net2.add_node(np.eye(2) * 4)
    net2.add_subnetwork(net1)
    self.assertIn(a, net2.nodes_set)
    self.assertIn(b, net2.nodes_set)
    e2 = net2.connect(c[0], a[1])
    e3 = net2.connect(c[1], b[1])
    net2.check_correct()
    for edge in [e1, e2, e3]:
      net2.contract(edge)
    result = net2.get_final_node()
    self.assertAllClose(result.get_tensor().numpy(), 48.0)

  def test_merge_networks(self):
    net1 = tensornetwork.TensorNetwork()
    net2 = tensornetwork.TensorNetwork()
    a = net1.add_node(np.eye(2) * 2)
    b = net1.add_node(np.eye(2) * 3)
    e1 = net1.connect(a[0], b[0])
    c = net2.add_node(np.eye(2) * 4)
    net3 = tensornetwork.TensorNetwork.merge_networks([net1, net2])
    self.assertIn(a, net3.nodes_set)
    self.assertIn(b, net3.nodes_set)
    e2 = net3.connect(c[0], a[1])
    e3 = net3.connect(c[1], b[1])
    net3.check_correct()
    for edge in [e1, e2, e3]:
      net3.contract(edge)
    result = net3.get_final_node()
    self.assertAllClose(result.get_tensor().numpy(), 48.0)

  def test_flatten_edges_between(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones((3, 4, 5)))
    b = net.add_node(np.ones((5, 4, 3)))
    net.connect(a[0], b[2])
    net.connect(a[1], b[1])
    net.connect(a[2], b[0])
    net.flatten_edges_between(a, b)
    net.check_correct()
    self.assertAllClose(a.get_tensor().numpy(), np.ones((60,)))
    self.assertAllClose(b.get_tensor().numpy(), np.ones((60,)))

  def test_flatten_edges_between_no_edges(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones((3)))
    b = net.add_node(np.ones((3)))
    self.assertEqual(net.flatten_edges_between(a, b), None)

  def test_flatten_all_edges(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones((3, 3, 5, 6, 2, 2)))
    b = net.add_node(np.ones((5, 6, 7)))
    c = net.add_node(np.ones((7,)))
    trace_edge1 = net.connect(a[0], a[1])
    trace_edge2 = net.connect(a[4], a[5])
    split_edge1 = net.connect(a[2], b[0])
    split_edge2 = net.connect(a[3], b[1])
    ok_edge = net.connect(b[2], c[0])
    flat_edges = net.flatten_all_edges()
    net.check_correct()
    self.assertEqual(len(flat_edges), 3)
    self.assertNotIn(trace_edge1, flat_edges)
    self.assertNotIn(trace_edge2, flat_edges)
    self.assertNotIn(split_edge1, flat_edges)
    self.assertNotIn(split_edge2, flat_edges)
    self.assertIn(ok_edge, flat_edges)

  def test_contract_between(self):
    net = tensornetwork.TensorNetwork()
    a_val = np.random.normal(size=(2, 3, 4, 5))
    b_val = np.random.normal(size=(3, 5, 4, 2))
    a = net.add_node(a_val)
    b = net.add_node(b_val)
    net.connect(a[0], b[3])
    net.connect(b[1], a[3])
    net.connect(a[1], b[0])
    edge_a = a[2]
    edge_b = b[2]
    c = net.contract_between(a, b, name="New Node")
    c.reorder_edges([edge_a, edge_b])
    net.check_correct()
    # Check expected values.
    a_flat = np.reshape(np.transpose(a_val, (2, 1, 0, 3)), (4, 30))
    b_flat = np.reshape(np.transpose(b_val, (2, 0, 3, 1)), (4, 30))
    final_val = np.matmul(a_flat, b_flat.T)
    self.assertAllClose(c.get_tensor().numpy(), final_val)
    self.assertEqual(c.name, "New Node")

  def test_contract_between_outer_product(self):
    net = tensornetwork.TensorNetwork()
    a_val = np.random.normal(size=(2, 3, 4))
    b_val = np.random.normal(size=(5, 6, 7))
    a = net.add_node(a_val)
    b = net.add_node(b_val)
    c = net.contract_between(a, b, allow_outer_product=True)
    self.assertEqual(c.get_tensor().shape, (2, 3, 4, 5, 6, 7))

  def test_contract_between_no_outer_product(self):
    net = tensornetwork.TensorNetwork()
    a_val = np.random.normal(size=(2, 3, 4))
    b_val = np.random.normal(size=(5, 6, 7))
    a = net.add_node(a_val)
    b = net.add_node(b_val)
    with self.assertRaises(ValueError):
      net.contract_between(a, b)

  def test_contract_between_trace_edges(self):
    net = tensornetwork.TensorNetwork()
    a_val = np.random.normal(size=(3, 3))
    final_val = np.trace(a_val)
    a = net.add_node(a_val)
    net.connect(a[0], a[1])
    b = net.contract_between(a, a)
    net.check_correct()
    self.assertAllClose(b.get_tensor().numpy(), final_val)

  def test_join_dangling(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.ones((3,)))
    b = net.add_node(np.ones((3,)))
    net.connect(a[0], b[0])
    net.check_correct()

  def test_dynamic_network_sizes(self):

    @tf.contrib.eager.defun
    def f(x, n):
      x_slice = x[:n]
      net = tensornetwork.TensorNetwork()
      n1 = net.add_node(x_slice)
      n2 = net.add_node(x_slice)
      e = net.connect(n1[0], n2[0])
      return net.contract(e).get_tensor()

    x = tf.ones(10)
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), 2.0)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), 3.0)

  def test_dynamic_network_sizes_flatten_standard(self):

    @tf.contrib.eager.defun
    def f(x, n):
      x_slice = x[..., :n]
      net = tensornetwork.TensorNetwork()
      n1 = net.add_node(x_slice)
      n2 = net.add_node(x_slice)
      net.connect(n1[0], n2[0])
      net.connect(n1[1], n2[1])
      net.connect(n1[2], n2[2])
      return net.contract(net.flatten_edges_between(n1, n2)).get_tensor()

    x = tf.ones((3, 4, 5))
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), 24.0)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), 36.0)

  def test_dynamic_network_sizes_flatten_trace(self):

    @tf.contrib.eager.defun
    def f(x, n):
      x_slice = x[..., :n]
      net = tensornetwork.TensorNetwork()
      n1 = net.add_node(x_slice)
      net.connect(n1[0], n1[2])
      net.connect(n1[1], n1[3])
      return net.contract(net.flatten_edges_between(n1, n1)).get_tensor()

    x = tf.ones((3, 4, 3, 4, 5))
    self.assertAllClose(f(x, tf.convert_to_tensor(2)), tf.ones((2,)) * 12)
    self.assertAllClose(f(x, tf.convert_to_tensor(3)), tf.ones((3,)) * 12)

  def test_split_node(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(tf.zeros((2, 3, 4, 5, 6)))
    left_edges = []
    for i in range(3):
      left_edges.append(a[i])
    right_edges = []
    for i in range(3, 5):
      right_edges.append(a[i])
    left, right, _ = net.split_node(a, left_edges, right_edges)
    net.check_correct()
    self.assertAllClose(left.get_tensor(), np.zeros((2, 3, 4, 24)))
    self.assertAllClose(right.get_tensor(), np.zeros((24, 5, 6)))

  def test_split_node_mixed_order(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(tf.zeros((2, 3, 4, 5, 6)))
    left_edges = []
    for i in [0, 2, 4]:
      left_edges.append(a[i])
    right_edges = []
    for i in [1, 3]:
      right_edges.append(a[i])
    left, right, _ = net.split_node(a, left_edges, right_edges)
    net.check_correct()
    self.assertAllClose(left.get_tensor(), np.zeros((2, 4, 6, 15)))
    self.assertAllClose(right.get_tensor(), np.zeros((15, 3, 5)))

  def test_split_node_full_svd(self):
    net = tensornetwork.TensorNetwork()
    random_matrix = np.random.rand(10, 10)
    unitary1, _, unitary2 = np.linalg.svd(random_matrix)
    singular_values = np.array(range(10))
    val = unitary1.dot(np.diag(singular_values).dot(unitary2.T))
    a = net.add_node(val)
    e1 = a[0]
    e2 = a[1]
    _, s, _, _, = net.split_node_full_svd(a, [e1], [e2])
    net.check_correct()
    self.assertAllClose(s.get_tensor(), np.diag(np.arange(9, -1, -1)))

  def test_batch_usage(self):
    def build_tensornetwork(tensors):
      net = tensornetwork.TensorNetwork()
      a = net.add_node(tensors[0])
      b = net.add_node(tensors[1])
      e = net.connect(a[0], b[0])
      return net.contract(e).get_tensor()

    tensors = [tf.ones((5, 10)), tf.ones((5, 10))]
    result = tf.map_fn(build_tensornetwork, tensors, dtype=tf.float32)
    self.assertAllClose(result, tf.ones(5) * 10)

  def test_weakref(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.eye(2))
    b = net.add_node(np.eye(2))
    e = net.connect(a[0], b[0])
    del a
    del b
    net.contract(e)
    with self.assertRaises(ValueError):
      e.node1
    with self.assertRaises(ValueError):
      e.node2

  def test_weakref_complex(self):
    net = tensornetwork.TensorNetwork()
    a = net.add_node(np.eye(2))
    b = net.add_node(np.eye(2))
    c = net.add_node(np.eye(2))
    e1 = net.connect(a[0], b[0])
    e2 = net.connect(b[1], c[0])
    net.contract(e1)
    net.contract(e2)
    # This won't raise an exception since we still have a referance to 'a'.
    e1.node1
    # This raises an exception since the intermediate tensor when doing
    # `net.contract(e2)` was garbage collected.
    with self.assertRaises(ValueError):
      e2.node1



if __name__ == "__main__":
  tf.test.main()

