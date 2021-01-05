import numpy as np
import tensorflow as tf
import torch
import pytest
from unittest.mock import patch
from collections import namedtuple
import h5py
import re
from tensornetwork.network_components import (Node, CopyNode, Edge,
                                              NodeCollection, AbstractNode,
                                              _remove_trace_edge, _remove_edges)
import tensornetwork as tn
from tensornetwork.backends.abstract_backend import AbstractBackend
from typing import Dict

string_type = h5py.special_dtype(vlen=str)

SingleNodeEdgeTensor = namedtuple('SingleNodeEdgeTensor', 'node edge tensor')
DoubleNodeEdgeTensor = namedtuple('DoubleNodeEdgeTensor',
                                  'node1 node2 edge1 edge12 tensor')

op_backend_dtype_values = [('numpy', np.float32, np.float32),
                           ('numpy', np.float64, np.float64),
                           ('numpy', np.complex64, np.complex64),
                           ('numpy', np.complex128, np.complex128),
                           ('pytorch', np.float32, torch.float32),
                           ('pytorch', np.float64, torch.float64),
                           ('tensorflow', np.float32, tf.float32),
                           ('tensorflow', np.float64, tf.float64),
                           ('tensorflow', np.complex64, tf.complex64),
                           ('tensorflow', np.complex128, tf.complex128),
                           ('jax', np.float32, np.float32),
                           ('jax', np.float64, np.float64),
                           ('jax', np.complex64, np.complex64),
                           ('jax', np.complex128, np.complex128)]


class TestNode(AbstractNode):

  def get_tensor(self):  #pylint: disable=useless-super-delegation
    return super().get_tensor()

  def set_tensor(self, tensor):  #pylint: disable=useless-super-delegation
    return super().set_tensor(tensor)

  @property
  def shape(self):
    return super().shape

  @property
  def tensor(self):
    return super().tensor

  #pylint: disable=no-member
  @tensor.setter
  def tensor(self, tensor):
    return super(TestNode, type(self)).tensor.fset(self, tensor)

  def _load_node(self, node_data):  # pylint: disable=useless-super-delegation
    return super()._load_node(node_data)

  def _save_node(self, node_group):  #pylint: disable=useless-super-delegation
    return super()._save_node(node_group)

  def copy(self, conjugate: bool = False) -> "TestNode":
    return TestNode()

  def to_serial_dict(self) -> Dict:
    return {}

  @classmethod
  def from_serial_dict(cls, serial_dict) -> "TestNode":
    return cls()


@pytest.fixture(name='single_node_edge')
def fixture_single_node_edge(backend):
  tensor = np.ones((1, 2, 2))
  node = Node(
      tensor=tensor,
      name="test_node",
      axis_names=["a", "b", "c"],
      backend=backend)
  edge = Edge(name="edge", node1=node, axis1=0)
  return SingleNodeEdgeTensor(node, edge, tensor)


@pytest.fixture(name='double_node_edge')
def fixture_double_node_edge(backend):
  tensor = np.ones((1, 2, 2))
  node1 = Node(
      tensor=tensor,
      name="test_node1",
      axis_names=["a", "b", "c"],
      backend=backend)
  node2 = Node(
      tensor=tensor,
      name="test_node2",
      axis_names=["a", "b", "c"],
      backend=backend)
  tn.connect(node1["b"], node2["b"])
  edge1 = Edge(name="edge", node1=node1, axis1=0)
  edge12 = Edge(name="edge", node1=node1, axis1=1, node2=node2, axis2=1)
  return DoubleNodeEdgeTensor(node1, node2, edge1, edge12, tensor)


@pytest.fixture(name='copy_node')
def fixture_copy_node(backend):
  return CopyNode(4, 2, "copier", ["a", "b", "c", "d"], backend=backend)


def test_node_initialize_numpy():
  tensor = np.ones((1, 2, 3))
  node = Node(
      tensor=tensor,
      name="test_node",
      axis_names=["a", "b", "c"],
      backend='numpy')
  np.testing.assert_allclose(node.tensor, tensor)
  assert node.name == 'test_node'
  assert len(node.edges) == 3
  assert isinstance(node.edges[0], Edge)
  assert node.axis_names == ["a", "b", "c"]


def test_node_initialize_tensorflow():
  tensor = tf.ones((1, 2, 3))
  node = Node(
      tensor=tensor,
      name="test_node",
      axis_names=["a", "b", "c"],
      backend='tensorflow')
  print(node.tensor)
  np.testing.assert_allclose(node.tensor, np.ones((1, 2, 3)))
  assert node.name == 'test_node'
  assert len(node.edges) == 3
  assert isinstance(node.edges[0], Edge)
  assert node.axis_names == ["a", "b", "c"]


def test_node_get_rank(single_node_edge):
  node = single_node_edge.node
  assert node.get_rank() == 3


def test_node_add_axis_names_raises_error_duplicate_names(single_node_edge):
  node = single_node_edge.node
  with pytest.raises(ValueError):
    node.add_axis_names(["A", "B", "A"])


def test_node_add_axis_names_raises_error_wrong_length(single_node_edge):
  node = single_node_edge.node
  with pytest.raises(ValueError):
    node.add_axis_names(["A", "B"])


def test_node_add_axis_names(single_node_edge):
  node = single_node_edge.node
  node.add_axis_names(["A", "B", "C"])
  assert node.axis_names == ["A", "B", "C"]


def test_node_add_edge_raises_error_mismatch_rank(single_node_edge):
  node = single_node_edge.node
  edge = single_node_edge.edge
  with pytest.raises(ValueError):
    node.add_edge(edge, axis=-1)
  edge = Edge(name="edge", node1=node, axis1=0)
  with pytest.raises(ValueError):
    node.add_edge(edge, axis=3)


def test_node_add_edge_raises_error_override(double_node_edge):
  node1 = double_node_edge.node1
  edge = double_node_edge.edge1
  with pytest.raises(ValueError):
    node1.add_edge(edge, axis=1)


def test_node_add_edge(single_node_edge):
  node = single_node_edge.node
  edge = single_node_edge.edge
  node.add_edge(edge, axis=0)
  assert node.edges[0] == edge


def test_node_get_tensor(single_node_edge):
  node = single_node_edge.node
  tensor = single_node_edge.tensor
  np.testing.assert_allclose(node.get_tensor(), tensor)


def test_node_set_tensor(single_node_edge):
  node = single_node_edge.node
  tensor2 = np.zeros((2, 4, 3, 2))
  node.set_tensor((tensor2))
  np.testing.assert_allclose(node.get_tensor(), tensor2)


def test_node_shape(single_node_edge):
  node = single_node_edge.node
  assert node.shape == (1, 2, 2)


def test_node_get_axis_number(single_node_edge):
  node = single_node_edge.node
  assert node.get_axis_number(1) == 1
  assert node.get_axis_number("b") == 1


def test_node_get_axis_number_raises_error_unknown(single_node_edge):
  node = single_node_edge.node
  with pytest.raises(ValueError):
    node.get_axis_number("d")


def test_node_get_dimension(single_node_edge):
  node = single_node_edge.node
  assert node.get_dimension(1) == 2
  assert node.get_dimension("b") == 2


def test_node_get_dimension_raises_error_mismatch_rank(single_node_edge):
  node = single_node_edge.node
  with pytest.raises(ValueError):
    node.get_dimension(-1)
  with pytest.raises(ValueError):
    node.get_dimension(3)


def test_node_get_edge(single_node_edge):
  node = single_node_edge.node
  edge = single_node_edge.edge
  node.add_edge(edge, axis=0)
  assert node.get_edge(0) == edge


def test_node_get_all_edges(single_node_edge):
  node = single_node_edge.node
  edge = single_node_edge.edge
  node.add_edge(edge, axis=0)
  assert len(node.get_all_edges()) == 3
  assert node.get_all_edges()[0] == edge


def test_node_get_all_nondangling(double_node_edge):
  node1 = double_node_edge.node1
  assert node1.get_all_nondangling() == {node1.get_all_edges()[1]}


def test_node_set_name(single_node_edge):
  node = single_node_edge.node
  node.set_name("new_name")
  assert node.name == "new_name"


def test_node_has_nondangling_edge_false(single_node_edge):
  node = single_node_edge.node
  assert not node.has_nondangling_edge()


def test_node_has_nondangling_edge_true(double_node_edge):
  node1 = double_node_edge.node1
  assert node1.has_nondangling_edge()


def test_node_reorder_edges(single_node_edge):
  node = single_node_edge.node
  e0 = node[0]
  e1 = node[1]
  e2 = node[2]
  node.reorder_edges([e1, e2, e0])
  assert node[0] == e1
  assert node[1] == e2
  assert node[2] == e0


def test_node_reorder_edges_raise_error_wrong_edges(single_node_edge):
  node = single_node_edge.node
  e0 = node[0]
  e1 = node[1]
  e2 = node[2]
  edge = Edge(name="edge", node1=node, axis1=0)
  with pytest.raises(ValueError) as e:
    node.reorder_edges([e0])
  assert "Missing edges that belong to node found:" in str(e.value)
  with pytest.raises(ValueError) as e:
    node.reorder_edges([e0, e1, e2, edge])
  assert "Additional edges that do not belong to node found:" in str(e.value)


def test_node_reorder_edges_raise_error_trace_edge(single_node_edge):
  node = single_node_edge.node
  e2 = tn.connect(node[1], node[2])
  e3 = node[0]
  with pytest.raises(ValueError) as e:
    node.reorder_edges([e2, e3])
  assert "Edge reordering does not support trace edges." in str(e.value)


def test_node_reorder_edges_raise_error_no_tensor(single_node_edge):
  node = single_node_edge.node
  e2 = tn.connect(node[1], node[2])
  e3 = node[0]
  del node._tensor
  with pytest.raises(AttributeError) as e:
    node.reorder_edges([e2, e3])
  assert "Please provide a valid tensor for this Node." in str(e.value)


def test_node_magic_getitem(single_node_edge):
  node = single_node_edge.node
  edge = single_node_edge.edge
  node.add_edge(edge, axis=0)
  assert node[0] == edge


def test_node_magic_getslice(single_node_edge):
  node = single_node_edge.node
  edge = single_node_edge.edge
  node.add_edge(edge, axis=0)
  assert node[:1][0] is edge
  assert len(node[None:None]) == 3
  assert len(node[0:3:2]) == 2


def test_node_magic_str(single_node_edge):
  node = single_node_edge.node
  assert str(node) == node.name


def test_node_magic_lt(double_node_edge):
  node1 = double_node_edge.node1
  node2 = double_node_edge.node2
  assert (node1 < node2) == (id(node1) < id(node2))


def test_node_magic_lt_raises_error_not_node(single_node_edge):
  node = single_node_edge.node
  with pytest.raises(ValueError):
    node < 0


def test_node_magic_matmul_raises_error_not_node(single_node_edge):
  node = single_node_edge.node
  with pytest.raises(TypeError):
    node @ 0


def test_node_magic_matmul_raises_error_no_tensor(single_node_edge):
  node = single_node_edge.node
  del node._tensor
  with pytest.raises(AttributeError):
    node @ node


def test_node_magic_matmul_raises_error_disabled_node(single_node_edge):
  node = single_node_edge.node
  node.is_disabled = True
  with pytest.raises(ValueError):
    node @ node


def test_node_edges_getter_raises_error_disabled_node(single_node_edge):
  node = single_node_edge.node
  node.is_disabled = True
  with pytest.raises(ValueError):
    node.edges


def test_node_edges_setter_raises_error_disabled_node(single_node_edge):
  node = single_node_edge.node
  node.is_disabled = True
  with pytest.raises(ValueError):
    node.edges = []


def test_node_magic_matmul_raises_error_different_network(single_node_edge):
  node = single_node_edge.node
  tensor = node.backend.convert_to_tensor(np.zeros((1, 2, 3)))
  node2 = Node(
      tensor=tensor,
      name="test",
      axis_names=["A", "B", "C"],
      backend=node.backend.name)
  with pytest.raises(ValueError):
    assert node @ node2


def test_node_magic_matmul(backend):

  tensor1 = np.ones((2, 3, 4, 5))
  tensor2 = 2 * np.ones((3, 5, 4, 2))
  node1 = tn.Node(tensor1, backend=backend)
  node2 = tn.Node(tensor2, backend=backend)
  tn.connect(node1[0], node2[3])
  tn.connect(node2[1], node1[3])
  tn.connect(node1[1], node2[0])
  actual = (node1 @ node2)
  expected = np.array([[60, 60, 60, 60], [60, 60, 60, 60], [60, 60, 60, 60],
                       [60, 60, 60, 60]])
  assert isinstance(actual, Node)
  np.testing.assert_allclose(actual.tensor, expected)


def test_between_node_add_op(backend):
  node1 = Node(tensor=np.array([[1, 2], [3, 4]]), backend=backend)
  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend=backend)
  node3 = Node(tensor=np.array([[1., 2.], [3., 4.]]), backend=backend)
  int_node = Node(tensor=np.array(2, dtype=np.int64), backend=backend)
  float_node = Node(tensor=np.array(2.5, dtype=np.float64), backend=backend)

  expected = np.array([[11, 12], [13, 14]])
  result = (node1 + node2).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node1.tensor.dtype == node2.tensor.dtype == result.dtype

  expected = np.array([[3, 4], [5, 6]])
  result = (node1 + int_node).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node1.tensor.dtype == int_node.tensor.dtype == result.dtype
  expected = np.array([[3, 4], [5, 6]])
  result = (int_node + node1).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node1.tensor.dtype == int_node.tensor.dtype == result.dtype

  expected = np.array([[3.5, 4.5], [5.5, 6.5]])
  result = (node3 + float_node).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node3.dtype == float_node.dtype == result.dtype
  expected = np.array([[3.5, 4.5], [5.5, 6.5]])
  result = (float_node + node3).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node3.dtype == float_node.dtype == result.dtype


def test_node_and_scalar_add_op(backend):
  node = Node(
      tensor=np.array([[1, 2], [3, 4]], dtype=np.int32), backend=backend)
  expected = np.array([[3, 4], [5, 6]])
  result = (node + 2).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node.tensor.dtype == result.dtype

  node = Node(
      tensor=np.array([[1, 2], [3, 4]], dtype=np.float32), backend=backend)
  expected = np.array([[3.5, 4.5], [5.5, 6.5]])
  result = (node + 2.5).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node.tensor.dtype == result.dtype


def test_between_node_sub_op(backend):
  node1 = Node(tensor=np.array([[1, 2], [3, 4]]), backend=backend)
  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend=backend)
  node3 = Node(tensor=np.array([[1., 2.], [3., 4.]]), backend=backend)
  int_node = Node(tensor=np.array(2, dtype=np.int64), backend=backend)
  float_node = Node(tensor=np.array(2.5, dtype=np.float64), backend=backend)

  expected = np.array([[-9, -8], [-7, -6]])
  result = (node1 - node2).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node1.tensor.dtype == node2.tensor.dtype == result.dtype

  expected = np.array([[-1, 0], [1, 2]])
  result = (node1 - int_node).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node1.tensor.dtype == int_node.tensor.dtype == result.dtype
  expected = np.array([[1, 0], [-1, -2]])
  result = (int_node - node1).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node1.tensor.dtype == int_node.tensor.dtype == result.dtype

  expected = np.array([[-1.5, -0.5], [0.5, 1.5]])
  result = (node3 - float_node).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node3.dtype == float_node.dtype == result.dtype
  expected = np.array([[1.5, 0.5], [-0.5, -1.5]])
  result = (float_node - node3).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node3.dtype == float_node.dtype == result.dtype


def test_node_and_scalar_sub_op(backend):
  node = Node(
      tensor=np.array([[1, 2], [3, 4]], dtype=np.int32), backend=backend)
  expected = np.array([[-1, 0], [1, 2]])
  result = (node - 2).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node.tensor.dtype == result.dtype
  node = Node(
      tensor=np.array([[1, 2], [3, 4]], dtype=np.float32), backend=backend)
  expected = np.array([[-1.5, -0.5], [0.5, 1.5]])
  result = (node - 2.5).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node.tensor.dtype == result.dtype


def test_between_node_mul_op(backend):
  node1 = Node(tensor=np.array([[1, 2], [3, 4]]), backend=backend)
  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend=backend)
  node3 = Node(tensor=np.array([[1., 2.], [3., 4.]]), backend=backend)
  int_node = Node(tensor=np.array(2, dtype=np.int64), backend=backend)
  float_node = Node(tensor=np.array(2.5, dtype=np.float64), backend=backend)

  expected = np.array([[10, 20], [30, 40]])
  result = (node1 * node2).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node1.tensor.dtype == node2.tensor.dtype == result.dtype

  expected = np.array([[2, 4], [6, 8]])
  result = (node1 * int_node).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node1.tensor.dtype == int_node.tensor.dtype == result.dtype
  result = (int_node * node1).tensor
  np.testing.assert_almost_equal(result, expected)

  expected = np.array([[2.5, 5], [7.5, 10]])
  result = (node3 * float_node).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node3.dtype == float_node.dtype == result.dtype
  result = (float_node * node3).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node3.dtype == float_node.dtype == result.dtype


def test_node_and_scalar_mul_op(backend):
  node = Node(
      tensor=np.array([[1, 2], [3, 4]], dtype=np.int32), backend=backend)
  expected = np.array([[2, 4], [6, 8]])
  result = (node * 2).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node.tensor.dtype == result.dtype

  node = Node(
      tensor=np.array([[1, 2], [3, 4]], dtype=np.float32), backend=backend)
  expected = np.array([[2.5, 5], [7.5, 10]])
  result = (node * 2.5).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node.tensor.dtype == result.dtype


@pytest.mark.parametrize("backend, npdtype, dtype", op_backend_dtype_values)
def test_between_node_truediv_op(backend, npdtype, dtype):
  node1 = Node(
      tensor=np.array([[1., 2.], [3., 4.]], dtype=npdtype), backend=backend)
  node2 = Node(
      tensor=np.array([[10., 10.], [10., 10.]], dtype=npdtype), backend=backend)
  expected = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=npdtype)
  result = (node1 / node2).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node1.tensor.dtype == node2.tensor.dtype == result.dtype == dtype


def test_between_node_div_op(backend):
  if backend == 'pytorch':
    pytest.skip("pytorch integer division no longer supported")

  node1 = Node(tensor=np.array([[1., 2.], [3., 4.]]), backend=backend)
  node2 = Node(tensor=np.array([[10., 10.], [10., 10.]]), backend=backend)
  node3 = Node(
      tensor=np.array([[1, 2], [3, 4]], dtype=np.int64), backend=backend)
  int_node = Node(tensor=np.array(2, dtype=np.int64), backend=backend)
  float_node = Node(tensor=np.array(2.5, dtype=np.float64), backend=backend)

  expected = np.array([[0.1, 0.2], [0.3, 0.4]])
  result = (node1 / node2).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node1.tensor.dtype == node2.tensor.dtype == result.dtype
  expected = np.array([[0.5, 1.], [1.5, 2.]])
  result = (node3 / int_node).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node3.tensor.dtype == 'int64'
  assert result.dtype == 'float64'

  expected = np.array([[2., 1.], [2 / 3, 0.5]])
  result = (int_node / node3).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node3.tensor.dtype == 'int64'
  assert result.dtype == 'float64'

  expected = np.array([[4., 4.], [4., 4.]])
  result = (node2 / float_node).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node2.dtype == float_node.dtype == result.dtype
  expected = np.array([[0.25, 0.25], [0.25, 0.25]])
  result = (float_node / node2).tensor
  np.testing.assert_almost_equal(result, expected)
  assert node2.dtype == float_node.dtype == result.dtype


@pytest.mark.parametrize("backend, npdtype, dtype", op_backend_dtype_values)
def test_node_and_scalar_div_op(backend, npdtype, dtype):
  node = Node(
      tensor=np.array([[5, 10], [15, 20]], dtype=npdtype), backend=backend)
  expected = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=npdtype)
  result = (node / 10).tensor
  np.testing.assert_almost_equal(result, expected)
  assert result.dtype == dtype
  assert node.tensor.dtype == dtype


def test_node_add_input_error():
  #pylint: disable=unused-variable
  #pytype: disable=unsupported-operands
  node1 = Node(tensor=2, backend='numpy')
  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend='numpy')

  del node1._tensor
  with pytest.raises(AttributeError):
    result = node1 + node2
  with pytest.raises(AttributeError):
    result = node2 + node1

  node1.tensor = 1
  node2 = 'str'
  copynode = tn.CopyNode(rank=4, dimension=3)
  with pytest.raises(TypeError):
    result = node1 + node2
  with pytest.raises(TypeError):
    result = node1 + copynode

  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend='pytorch')
  with pytest.raises(TypeError):
    result = node1 + node2
  #pytype: enable=unsupported-operands


def test_node_sub_input_error():
  #pylint: disable=unused-variable
  #pytype: disable=unsupported-operands
  node1 = Node(tensor=2, backend='numpy')
  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend='numpy')

  del node1._tensor
  with pytest.raises(AttributeError):
    result = node1 - node2
    result = node2 - node1

  node1.tensor = 1
  node2 = 'str'
  copynode = tn.CopyNode(rank=4, dimension=3)
  with pytest.raises(TypeError):
    result = node1 - node2
  with pytest.raises(TypeError):
    result = node1 - copynode

  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend='pytorch')
  with pytest.raises(TypeError):
    result = node1 - node2
  #pytype: enable=unsupported-operands


def test_node_mul_input_error():
  #pylint: disable=unused-variable
  #pytype: disable=unsupported-operands
  node1 = Node(tensor=2, backend='numpy')
  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend='numpy')

  del node1._tensor
  with pytest.raises(AttributeError):
    result = node1 * node2
    result = node2 * node1

  node1.tensor = 1
  node2 = 'str'
  copynode = tn.CopyNode(rank=4, dimension=3)
  with pytest.raises(TypeError):
    result = node1 * node2
  with pytest.raises(TypeError):
    result = node1 * copynode

  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend='pytorch')
  with pytest.raises(TypeError):
    result = node1 * node2
  #pytype: enable=unsupported-operands


def test_node_div_input_error():
  #pylint: disable=unused-variable
  #pytype: disable=unsupported-operands
  node1 = Node(tensor=2, backend='numpy')
  node1 = Node(tensor=2, backend='numpy')
  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend='numpy')

  del node1._tensor
  with pytest.raises(AttributeError):
    result = node1 / node2
  with pytest.raises(AttributeError):
    result = node2 / node1

  node1.tensor = 1
  node2 = 'str'
  copynode = tn.CopyNode(rank=4, dimension=3)
  with pytest.raises(TypeError):
    result = node1 / node2
  with pytest.raises(TypeError):
    result = node1 / copynode

  node2 = Node(tensor=np.array([[10, 10], [10, 10]]), backend='pytorch')
  with pytest.raises(TypeError):
    result = node1 / node2
  #pytype: enable=unsupported-operands


def test_node_save_structure(tmp_path, single_node_edge):
  node = single_node_edge.node
  with h5py.File(tmp_path / 'nodes', 'w') as node_file:
    node_group = node_file.create_group('test_node')
    node._save_node(node_group)
    assert set(list(node_file.keys())) == {"test_node"}
    assert set(list(node_file['test_node'])) == {
        "tensor", 'backend', 'name', 'edges', 'shape', 'axis_names', "type"
    }


def test_node_save_data(tmp_path, single_node_edge):
  node = single_node_edge.node
  with h5py.File(tmp_path / 'nodes', 'w') as node_file:
    node_group = node_file.create_group('test_node')
    node._save_node(node_group)
    np.testing.assert_allclose(node_file['test_node/tensor'][()], node.tensor)
    assert node_file['test_node/backend'][()] == node.backend.name
    assert node_file['test_node/type'][()] == type(node).__name__
    assert node_file['test_node/name'][()] == node.name
    assert set(node_file['test_node/shape'][()]) == set(node.shape)
    assert set(node_file['test_node/axis_names'][()]) == set(node.axis_names)
    assert (set(node_file['test_node/edges'][()]) == set(
        edge.name for edge in node.edges))


def test_node_load(tmp_path, single_node_edge):
  node = single_node_edge.node
  with h5py.File(tmp_path / 'node', 'w') as node_file:
    node_group = node_file.create_group('node_data')
    node_group.create_dataset('tensor', data=node._tensor)
    node_group.create_dataset('backend', data=node.backend.name)
    node_group.create_dataset('name', data=node.name)
    node_group.create_dataset('shape', data=node.shape)
    node_group.create_dataset(
        'axis_names',
        data=np.array(node.axis_names, dtype=object),
        dtype=string_type)
    node_group.create_dataset(
        'edges',
        data=np.array([edge.name for edge in node.edges], dtype=object),
        dtype=string_type)

    loaded_node = Node._load_node(node_data=node_file["node_data/"])
    assert loaded_node.name == node.name
    assert loaded_node.backend.name == node.backend.name
    assert set(loaded_node.axis_names) == set(node.axis_names)
    assert (set(edge.name for edge in loaded_node.edges) == set(
        edge.name for edge in node.edges))
    np.testing.assert_allclose(loaded_node.tensor, node.tensor)


def test_copy_node_init(copy_node):
  assert copy_node.rank == 4
  assert copy_node.dimension == 2
  assert copy_node.name == "copier"
  assert copy_node.axis_names == ["a", "b", "c", "d"]
  assert copy_node._tensor is None


def test_copy_node_shape(copy_node):
  assert copy_node.shape == (2, 2, 2, 2)


def test_copy_node_tensor(copy_node):
  expected = np.array(([[[[1, 0], [0, 0]], [[0, 0], [0, 0]]],
                        [[[0, 0], [0, 0]], [[0, 0], [0, 1]]]]))
  np.testing.assert_allclose(copy_node.get_tensor(), expected)
  np.testing.assert_allclose(copy_node.tensor, expected)
  np.testing.assert_allclose(copy_node._tensor, expected)


def test_copy_node_make_copy_tensor(copy_node):
  expected = np.array(([[[[1, 0], [0, 0]], [[0, 0], [0, 0]]],
                        [[[0, 0], [0, 0]], [[0, 0], [0, 1]]]]))
  np.testing.assert_allclose(
      copy_node.make_copy_tensor(4, 2, dtype=np.int64), expected)


def test_copy_node_set_tensor(copy_node):
  expected = np.ones((2, 3, 4))
  copy_node.set_tensor(expected)
  np.testing.assert_allclose(copy_node.get_tensor(), expected)
  np.testing.assert_allclose(copy_node.tensor, expected)
  np.testing.assert_allclose(copy_node._tensor, expected)


def test_copy_node_set_tensor_property(copy_node):
  expected = np.ones((2, 3, 4))
  copy_node.tensor = expected
  np.testing.assert_allclose(copy_node.get_tensor(), expected)
  np.testing.assert_allclose(copy_node.tensor, expected)
  np.testing.assert_allclose(copy_node._tensor, expected)


def test_copy_node_save_structure(tmp_path, backend):
  node = tn.CopyNode(
      rank=4,
      dimension=3,
      name='copier',
      axis_names=[str(n) for n in range(4)],
      backend=backend)
  with h5py.File(tmp_path / 'nodes', 'w') as node_file:
    node_group = node_file.create_group('test_node')
    node._save_node(node_group)
    assert set(list(node_file.keys())) == {"test_node"}
    assert set(list(node_file['test_node'])) == {
        'name', 'edges', 'backend', 'shape', 'axis_names', 'copy_node_dtype',
        "type"
    }


def test_copy_node_save_data(tmp_path, backend):
  node = tn.CopyNode(
      rank=4,
      dimension=3,
      name='copier',
      axis_names=[str(n) for n in range(4)],
      backend=backend)
  with h5py.File(tmp_path / 'nodes', 'w') as node_file:
    node_group = node_file.create_group('copier')
    node._save_node(node_group)
    assert node_file['copier/backend'][()] == node.backend.name
    assert node_file['copier/type'][()] == type(node).__name__
    assert node_file['copier/name'][()] == node.name
    assert node_file['copier/copy_node_dtype'][()] == np.dtype(
        node.copy_node_dtype).name
    assert set(node_file['copier/shape'][()]) == set(node.shape)
    assert set(node_file['copier/axis_names'][()]) == set(node.axis_names)
    assert (set(node_file['copier/edges'][()]) == set(
        edge.name for edge in node.edges))


def test_copy_node_load(tmp_path, backend):
  node = tn.CopyNode(
      rank=4,
      dimension=3,
      name='copier',
      axis_names=[str(n) for n in range(4)],
      backend=backend)
  with h5py.File(tmp_path / 'node', 'w') as node_file:
    node_group = node_file.create_group('node_data')
    node_group.create_dataset('backend', data=node.backend.name)
    node_group.create_dataset(
        'copy_node_dtype', data=np.dtype(node.copy_node_dtype).name)
    node_group.create_dataset('name', data=node.name)
    node_group.create_dataset('shape', data=node.shape)
    node_group.create_dataset(
        'axis_names',
        data=np.array(node.axis_names, dtype=object),
        dtype=string_type)
    node_group.create_dataset(
        'edges',
        data=np.array([edge.name for edge in node.edges], dtype=object),
        dtype=string_type)

    loaded_node = CopyNode._load_node(node_data=node_file["node_data/"])
    assert loaded_node.name == node.name
    assert set(loaded_node.axis_names) == set(node.axis_names)
    assert (set(edge.name for edge in loaded_node.edges) == set(
        edge.name for edge in node.edges))
    assert loaded_node.get_dimension(axis=1) == node.get_dimension(axis=1)
    assert loaded_node.get_rank() == node.get_rank()
    assert loaded_node.shape == node.shape
    assert loaded_node.copy_node_dtype == node.copy_node_dtype


def test_edge_initialize_dangling(single_node_edge):
  node = single_node_edge.node
  edge = single_node_edge.edge
  assert edge.name == "edge"
  assert edge.node1 == node
  assert edge.axis1 == 0
  assert edge.node2 is None
  assert edge.axis2 is None
  assert edge.is_dangling() is True


def test_edge_initialize_nondangling(double_node_edge):
  node1 = double_node_edge.node1
  node2 = double_node_edge.node2
  edge = double_node_edge.edge12
  assert edge.name == "edge"
  assert edge.node1 == node1
  assert edge.axis1 == 1
  assert edge.node2 == node2
  assert edge.axis2 == 1
  assert edge.is_dangling() is False


def test_edge_initialize_raises_error_faulty_arguments(double_node_edge):
  node1 = double_node_edge.node1
  node2 = double_node_edge.node2
  with pytest.raises(ValueError):
    Edge(name="edge", node1=node1, node2=node2, axis1=0)
  with pytest.raises(ValueError):
    Edge(name="edge", node1=node1, axis1=0, axis2=0)


def test_edge_get_nodes_single(single_node_edge):
  node = single_node_edge.node
  edge = single_node_edge.edge
  assert edge.get_nodes() == [node, None]


def test_edge_get_nodes_double(double_node_edge):
  node1 = double_node_edge.node1
  node2 = double_node_edge.node2
  edge = double_node_edge.edge12
  assert edge.get_nodes() == [node1, node2]


def test_edge_update_axis(double_node_edge):
  node1 = double_node_edge.node1
  node2 = double_node_edge.node2
  edge = double_node_edge.edge1
  edge.update_axis(old_axis=0, old_node=node1, new_axis=2, new_node=node2)
  assert edge.node1 == node2
  assert edge.axis1 == 2


def test_edge_update_axis_raises_error_old_node(double_node_edge):
  node2 = double_node_edge.node2
  edge = double_node_edge.edge1
  with pytest.raises(ValueError):
    edge.update_axis(old_axis=0, old_node=node2, new_axis=2, new_node=node2)


def test_edge_node1_property(single_node_edge):
  node = single_node_edge.node
  edge = single_node_edge.edge
  assert edge.node1 == node


def test_edge_node1_setter(double_node_edge):
  node2 = double_node_edge.node2
  edge = double_node_edge.edge1
  edge.node1 = node2
  assert edge.node1 == node2


def test_edge_node2_property(double_node_edge):
  node2 = double_node_edge.node2
  edge1 = double_node_edge.edge1
  edge12 = double_node_edge.edge12
  assert edge1.node2 is None
  assert edge12.node2 == node2


def test_edge_node2_setter(double_node_edge):
  node1 = double_node_edge.node1
  edge12 = double_node_edge.edge12
  edge12.node2 = node1
  assert edge12.node2 == node1


def test_edge_dimension(single_node_edge):
  edge = single_node_edge.edge
  assert edge.dimension == 1


def test_edge_is_dangling(double_node_edge):
  edge1 = double_node_edge.edge1
  edge12 = double_node_edge.edge12
  assert edge1.is_dangling()
  assert not edge12.is_dangling()


def test_edge_is_trace_true(single_node_edge):
  node = single_node_edge.node
  edge = Edge(name="edge", node1=node, axis1=1, node2=node, axis2=2)
  assert edge.is_trace()


def test_edge_is_trace_false(double_node_edge):
  edge1 = double_node_edge.edge1
  edge12 = double_node_edge.edge12
  assert not edge1.is_trace()
  assert not edge12.is_trace()


def test_edge_is_being_used_true(double_node_edge):
  node1 = double_node_edge.node1
  assert node1.get_all_edges()[0].is_being_used()
  assert node1.get_all_edges()[1].is_being_used()


def test_edge_is_being_used_false(single_node_edge):
  node = single_node_edge.node
  edge2 = Edge(name="edge", node1=node, axis1=0)
  assert not edge2.is_being_used()


def test_edge_set_name(single_node_edge):
  edge = single_node_edge.edge
  edge.set_name('new_name')
  assert edge.name == 'new_name'


def test_edge_magic_xor(double_node_edge):
  node1 = double_node_edge.node1
  node2 = double_node_edge.node2
  edge1 = Edge(name="edge1", node1=node1, axis1=2)
  edge2 = Edge(name="edge2", node1=node2, axis1=2)
  edge = edge1 ^ edge2
  assert edge.node1 == node1
  assert edge.node2 == node2


def test_edge_magic_lt_raise_error_type(single_node_edge):
  edge = single_node_edge.edge
  with pytest.raises(TypeError):
    assert edge < 0


def test_edge_magic_str(single_node_edge):
  edge = single_node_edge.edge
  assert str(edge) == edge.name


def test_edge_node_save_structure(tmp_path, double_node_edge):
  edge12 = double_node_edge.edge12
  with h5py.File(tmp_path / 'edges', 'w') as edge_file:
    edge_group = edge_file.create_group('edge')
    edge12._save_edge(edge_group)
    assert set(list(
        edge_group.keys())) == {"axis1", "node1", "axis2", "node2", "name"}


def test_edge_node_save_data(tmp_path, double_node_edge):
  edge = double_node_edge.edge12
  with h5py.File(tmp_path / 'edges', 'w') as edge_file:
    edge_group = edge_file.create_group('edge')
    edge._save_edge(edge_group)
    assert edge_file['edge/name'][()] == edge.name
    assert edge_file['edge/node1'][()] == edge.node1.name
    assert edge_file['edge/node2'][()] == edge.node2.name
    assert edge_file['edge/axis1'][()] == edge.axis1
    assert edge_file['edge/axis2'][()] == edge.axis2


def test_edge_load(backend, tmp_path, double_node_edge):
  edge = double_node_edge.edge12

  with h5py.File(tmp_path / 'edge', 'w') as edge_file:
    edge_group = edge_file.create_group('edge_data')
    edge_group.create_dataset('name', data=edge.name)
    edge_group.create_dataset('node1', data=edge.node1.name)
    edge_group.create_dataset('node2', data=edge.node2.name)
    edge_group.create_dataset('axis1', data=edge.axis1)
    edge_group.create_dataset('axis2', data=edge.axis2)

    ten = np.ones((1, 2, 2))
    node1 = Node(
        tensor=2 * ten,
        name="test_node1",
        axis_names=["a", "b", "c"],
        backend=backend)
    node2 = Node(
        tensor=ten,
        name="test_node2",
        axis_names=["a", "b", "c"],
        backend=backend)
    loaded_edge = Edge._load_edge(edge_group, {
        node1.name: node1,
        node2.name: node2
    })
    assert loaded_edge.name == edge.name
    assert loaded_edge.node1.name == edge.node1.name
    assert loaded_edge.node2.name == edge.node2.name
    assert loaded_edge.axis1 == edge.axis1
    assert loaded_edge.axis2 == edge.axis2
    np.testing.assert_allclose(loaded_edge.node1.tensor, node1.tensor)
    np.testing.assert_allclose(loaded_edge.node2.tensor, node2.tensor)


def test_disabled_edge_access(backend):

  n1 = Node(np.random.rand(2), backend=backend)
  n2 = Node(np.random.rand(2), backend=backend)
  e = n1[0] ^ n2[0]
  e.disable()
  with pytest.raises(ValueError):
    e.node1
  with pytest.raises(ValueError):
    e.node2
  with pytest.raises(ValueError):
    e.axis1
  with pytest.raises(ValueError):
    e.axis2


def test_disabled_edge_setter(backend):
  n1 = Node(np.random.rand(2), backend=backend)
  n2 = Node(np.random.rand(2), backend=backend)
  e = n1[0] ^ n2[0]
  e.disable()
  with pytest.raises(ValueError):
    e.node1 = None
  with pytest.raises(ValueError):
    e.node2 = None
  with pytest.raises(ValueError):
    e.axis1 = None
  with pytest.raises(ValueError):
    e.axis2 = None


def test_disconnect(backend):
  n1 = Node(np.random.rand(2), backend=backend)
  n2 = Node(np.random.rand(2), backend=backend)
  e = n1[0] ^ n2[0]
  assert not e.is_dangling()
  dangling_edge_1, dangling_edge_2 = e.disconnect('left_name', 'right_name')
  tn.check_correct([n1, n2], False)
  assert dangling_edge_1.is_dangling()
  assert dangling_edge_2.is_dangling()
  assert n1[0].is_dangling()
  assert n2[0].is_dangling()
  assert n1[0].name == 'left_name'
  assert n2[0].name == 'right_name'
  assert n1.get_edge(0) == dangling_edge_1
  assert n2.get_edge(0) == dangling_edge_2


def test_disconnect_dangling_edge_value_error(backend):
  a = tn.Node(np.eye(2), backend=backend)
  with pytest.raises(ValueError):
    a[0].disconnect()


def test_broken_edge_contraction(backend):
  n1 = Node(np.random.rand(2), backend=backend)
  n2 = Node(np.random.rand(2), backend=backend)
  e = n1[0] ^ n2[0]
  e.disconnect('left_name', 'right_name')
  with pytest.raises(ValueError):
    n1 @ n2


def test_disconnect_magicmethod(backend):
  n1 = Node(np.random.rand(2), backend=backend)
  n2 = Node(np.random.rand(2), backend=backend)
  _ = n1[0] ^ n2[0]
  n1[0] | n2[0]

  assert n1[0].is_dangling()
  assert n2[0].is_dangling()


def test_broken_edge_contraction_magicmethod(backend):
  n1 = Node(np.random.rand(2), backend=backend)
  n2 = Node(np.random.rand(2), backend=backend)
  _ = n1[0] ^ n2[0]
  n1[0] | n2[0]
  with pytest.raises(ValueError):
    n1 @ n2


def test_save_nodes_raise(backend, tmp_path):
  nodes = [
      Node(
          np.random.rand(2, 2, 2, 2),
          backend=backend,
          name='Node{}'.format(n),
          axis_names=[
              'node{}_1'.format(n), 'node_{}_2'.format(n),
              'node_{}_3'.format(n), 'node_{}_4'.format(n)
          ]) for n in range(4)
  ]
  _ = [nodes[n][0] ^ nodes[n + 1][1] for n in range(3)]
  with pytest.raises(ValueError):
    tn.save_nodes([nodes[0], nodes[1]], tmp_path / 'test_file_save_nodes')


def test_save_nodes_raise_2(backend, tmp_path):
  node = Node(
      np.random.rand(2, 2, 2, 2),
      backend=backend,
      name='Node',
      axis_names=['node_1', 'node_2', 'node_3', 'node_4'])

  with pytest.raises(ValueError):
    tn.save_nodes([node, node], tmp_path / 'test_file_save_nodes')


def test_save_load_nodes(backend, tmp_path):
  nodes = [
      Node(
          np.random.rand(2, 2, 2, 2),
          backend=backend,
          name='Node{}'.format(n),
          axis_names=[
              'node{}_1'.format(n), 'node{}_2'.format(n), 'node{}_3'.format(n),
              'node{}_4'.format(n)
          ]) for n in range(4)
  ]

  nodes[0][0] ^ nodes[1][1]
  nodes[2][1] ^ nodes[2][2]

  tn.save_nodes(nodes, tmp_path / 'test_file_save_nodes')

  loaded_nodes = tn.load_nodes(tmp_path / 'test_file_save_nodes')
  for n, node in enumerate(nodes):
    assert node.name == loaded_nodes[n].name
    assert node.axis_names == loaded_nodes[n].axis_names
    assert node.backend.name == loaded_nodes[n].backend.name
    np.testing.assert_allclose(node.tensor, loaded_nodes[n].tensor)

  res = nodes[0] @ nodes[1]
  loaded_res = loaded_nodes[0] @ loaded_nodes[1]
  np.testing.assert_allclose(res.tensor, loaded_res.tensor)

  trace = tn.contract_trace_edges(nodes[2])
  loaded_trace = tn.contract_trace_edges(loaded_nodes[2])
  np.testing.assert_allclose(trace.tensor, loaded_trace.tensor)


def test_add_to_node_collection_list():
  container = []
  with NodeCollection(container):
    a = Node(np.eye(2))
    b = Node(np.eye(3))

  assert container == [a, b]


def test_add_to_node_collection_set():
  container = set()
  with NodeCollection(container):
    a = Node(np.eye(2))
    b = Node(np.eye(3))

  assert container == {a, b}


def test_copy_node_add_to_node_collection():
  container = set()
  with NodeCollection(container):
    a = tn.CopyNode(
        rank=4,
        dimension=3,
        name='copier1',
        axis_names=[str(n) for n in range(4)])
    b = tn.CopyNode(
        rank=2,
        dimension=3,
        name='copier2',
        axis_names=[str(n) for n in range(2)])
  assert container == {a, b}


def test_add_to_node_collection_nested():
  container1 = set()
  container2 = set()
  with NodeCollection(container1):
    with NodeCollection(container2):
      a = Node(np.eye(2))
      b = Node(np.eye(3))

  assert container1 == set()
  assert container2 == {a, b}


def test_repr_for_Nodes_and_Edges(double_node_edge):
  node1 = repr(double_node_edge.node1)
  node1 = re.sub(r"\s", "", node1)
  node1 = re.sub(r"\s", "", node1)
  node2 = repr(double_node_edge.node2)
  node2 = re.sub(r"\s", "", node2)
  node2 = re.sub(r"\s", "", node2)
  assert "test_node1" in str(node1)
  assert "[[[1.,1.],[1.,1.]]]" in str(node1) and str(node2)
  assert "Edge(DanglingEdge)[0]" in str(node1) and str(node2)
  assert "Edge('test_node1'[1]->'test_node2'[1])" in str(node1) and str(node2)
  assert "Edge(DanglingEdge)[2]" in str(node1) and str(node2)


def test_base_node_name_list_throws_error():
  with pytest.raises(TypeError,):
    #pylint: disable=line-too-long
    TestNode(name=["A"], axis_names=['a', 'b'])  # pytype: disable=wrong-arg-types


def test_base_node_name_int_throws_error():
  with pytest.raises(TypeError):
    TestNode(name=1, axis_names=['a', 'b'])  # pytype: disable=wrong-arg-types


def test_base_node_axis_names_int_throws_error():
  with pytest.raises(TypeError):
    TestNode(axis_names=[0, 1])  # pytype: disable=wrong-arg-types


def test_base_node_no_axis_names_no_shapes_throws_error():
  with pytest.raises(ValueError):
    TestNode(name='a')


def test_node_add_axis_names_int_throws_error():
  n1 = Node(np.eye(2), axis_names=['a', 'b'])
  with pytest.raises(TypeError):
    n1.add_axis_names([0, 1])  # pytype: disable=wrong-arg-types


def test_node_axis_names_setter_throws_shape_large_mismatch_error():
  n1 = Node(np.eye(2), axis_names=['a', 'b'])
  with pytest.raises(ValueError):
    n1.axis_names = ['a', 'b', 'c']


def test_node_axis_names_setter_throws_shape_small_mismatch_error():
  n1 = Node(np.eye(2), axis_names=['a', 'b'])
  with pytest.raises(ValueError):
    n1.axis_names = ['a']


def test_node_axis_names_setter_throws_value_error():
  n1 = Node(np.eye(2), axis_names=['a', 'b'])
  with pytest.raises(TypeError):
    n1.axis_names = [0, 1]


def test_node_dtype(backend):
  n1 = Node(np.random.rand(2), backend=backend)
  assert n1.dtype == n1.tensor.dtype


@pytest.mark.parametrize("name", [1, ['1']])
def test_node_set_name_raises_type_error(backend, name):
  n1 = Node(np.random.rand(2), backend=backend)
  with pytest.raises(TypeError):
    n1.set_name(name)


@pytest.mark.parametrize("name", [1, ['1']])
def test_node_name_setter_raises_type_error(backend, name):
  n1 = Node(np.random.rand(2), backend=backend)
  with pytest.raises(TypeError):
    n1.name = name


def test_base_node_get_tensor():
  n1 = TestNode(name="n1", axis_names=['a'], shape=(1,))
  assert n1.get_tensor() is None


def test_base_node_set_tensor():
  n1 = TestNode(name="n1", axis_names=['a'], shape=(1,))
  assert n1.set_tensor(np.random.rand(2)) is None
  assert n1.tensor is None


def test_base_node_shape():
  n1 = TestNode(name="n1", axis_names=['a'], shape=(1,))
  n1._shape = None
  with pytest.raises(ValueError):
    n1.shape


def test_base_node_tensor_getter():
  n1 = TestNode(name="n1", axis_names=['a'], shape=(1,))
  assert n1.tensor is None


def test_base_node_tensor_setter():
  n1 = TestNode(name="n1", axis_names=['a'], shape=(1,))
  n1.tensor = np.random.rand(2)
  assert n1.tensor is None


def test_node_has_dangling_edge_false(double_node_edge):
  node1 = double_node_edge.node1
  node2 = double_node_edge.node2
  tn.connect(node1["a"], node2["a"])
  tn.connect(node1["c"], node2["c"])
  assert not node1.has_dangling_edge()


def test_node_has_dangling_edge_true(single_node_edge):
  assert single_node_edge.node.has_dangling_edge()


def test_node_get_item(single_node_edge):
  node = single_node_edge.node
  edge = single_node_edge.edge
  node.add_edge(edge, axis=0)
  assert node[0] == edge
  assert edge in node[0:2]


def test_node_disabled_disabled_throws_error(single_node_edge):
  node = single_node_edge.node
  node.is_disabled = True
  with pytest.raises(ValueError):
    node.disable()


def test_node_disabled_shape_throws_error(single_node_edge):
  node = single_node_edge.node
  node.is_disabled = True
  with pytest.raises(ValueError):
    node.shape


def test_copy_node_get_partners_with_trace(backend):
  node1 = CopyNode(4, 2, backend=backend)
  node2 = Node(np.random.rand(2, 2), backend=backend, name="node2")
  tn.connect(node1[0], node1[1])
  tn.connect(node1[2], node2[0])
  tn.connect(node1[3], node2[1])
  assert node1.get_partners() == {node2: {0, 1}}


@pytest.mark.parametrize("name", [1, ['1']])
def test_edge_name_throws_type_error(single_node_edge, name):
  with pytest.raises(TypeError):
    Edge(node1=single_node_edge.node, axis1=0, name=name)


def test_edge_name_setter_disabled_throws_error(single_node_edge):
  edge = Edge(node1=single_node_edge.node, axis1=0)
  edge.is_disabled = True
  with pytest.raises(ValueError):
    edge.name = 'edge'


def test_edge_name_getter_disabled_throws_error(single_node_edge):
  edge = Edge(node1=single_node_edge.node, axis1=0)
  edge.is_disabled = True
  with pytest.raises(ValueError):
    edge.name


@pytest.mark.parametrize("name", [1, ['1']])
def test_edge_name_setter_throws_type_error(single_node_edge, name):
  edge = Edge(node1=single_node_edge.node, axis1=0)
  with pytest.raises(TypeError):
    edge.name = name


def test_edge_node1_throws_value_error(single_node_edge):
  edge = Edge(node1=single_node_edge.node, axis1=0, name="edge")
  edge._nodes[0] = None
  err_msg = "node1 for edge 'edge' no longer exists."
  with pytest.raises(ValueError, match=err_msg):
    edge.node1


def test_edge_node2_throws_value_error(single_node_edge):
  edge = tn.connect(single_node_edge.node[1], single_node_edge.node[2])
  edge.name = 'edge'
  edge._nodes[1] = None
  err_msg = "node2 for edge 'edge' no longer exists."
  with pytest.raises(ValueError, match=err_msg):
    edge.node2


@pytest.mark.parametrize("name", [1, ['1']])
def test_edge_set_name_throws_type_error(single_node_edge, name):
  edge = Edge(node1=single_node_edge.node, axis1=0)
  with pytest.raises(TypeError):
    edge.set_name(name)


@patch.object(Edge, "name", None)
def test_edge_str(single_node_edge):
  single_node_edge.edge.name = None
  assert str(single_node_edge.edge) == "__unnamed_edge__"


def test_get_all_dangling_single_node(single_node_edge):
  node = single_node_edge.node
  assert set(tn.get_all_dangling({node})) == set(node.edges)


def test_get_all_dangling_double_node(double_node_edge):
  node1 = double_node_edge.node1
  node2 = double_node_edge.node2
  assert set(tn.get_all_dangling(
      {node1, node2})) == {node1[0], node1[2], node2[0], node2[2]}


def test_flatten_edges_different_backend_raises_value_error(single_node_edge):
  node1 = single_node_edge.node
  node2 = tn.Node(np.random.rand(2, 2, 2))
  node2.backend = AbstractBackend()
  with pytest.raises(ValueError):
    tn.flatten_edges(node1.get_all_edges() + node2.get_all_edges())


def test_split_edge_trivial(single_node_edge):
  edge = single_node_edge.edge
  assert tn.split_edge(edge, (1,)) == [edge]


def test_split_edge_different_backend_raises_value_error(single_node_edge):
  if single_node_edge.node.backend.name == "numpy":
    pytest.skip("numpy comparing to all the others")
  node1 = single_node_edge.node
  node2 = tn.Node(np.random.rand(2, 2, 2), backend="numpy")
  edge = tn.connect(node1[1], node2[1])
  with pytest.raises(ValueError, match="Not all backends are the same."):
    tn.split_edge(edge, (2, 1))


def test_slice_edge_different_backend_raises_value_error(single_node_edge):
  if single_node_edge.node.backend.name == "numpy":
    pytest.skip("numpy comparing to all the others")
  node1 = single_node_edge.node
  node2 = tn.Node(np.random.rand(2, 2, 2), backend="numpy")
  edge = tn.connect(node1[1], node2[1])
  with pytest.raises(ValueError, match="Not all backends are the same."):
    tn.slice_edge(edge, 0, 1)


def test_slice_edge_trace_edge(backend):
  node = Node(np.arange(9).reshape(3, 3), backend=backend)
  edge = tn.connect(node[0], node[1])
  new_edge = tn.slice_edge(edge, start_index=1, length=2)

  assert new_edge.node1 == node
  assert new_edge.node2 == node
  assert new_edge.axis1 == 0
  assert new_edge.axis2 == 1
  assert new_edge.dimension == 2

  expected_tensor = np.array([[4, 5], [7, 8]])
  np.testing.assert_allclose(expected_tensor, node.get_tensor())


def test_slice_edge_dangling_edge(backend):
  node = Node(np.arange(9).reshape(3, 3), backend=backend)
  edge = node[0]
  new_edge = tn.slice_edge(edge, start_index=1, length=2)

  assert new_edge.node1 == node
  assert new_edge.node2 is None
  assert new_edge.axis1 == 0
  assert new_edge.axis2 is None
  assert new_edge.dimension == 2

  expected_tensor = np.array([[3, 4, 5], [6, 7, 8]])
  np.testing.assert_allclose(expected_tensor, node.get_tensor())


def test_slice_edge_standard_edge(backend):
  node_1 = Node(np.arange(9).reshape(3, 3), backend=backend)
  node_2 = Node(np.arange(12).reshape(3, 4), backend=backend)
  edge = tn.connect(node_1[1], node_2[0])
  new_edge = tn.slice_edge(edge, start_index=1, length=2)

  assert new_edge.node1 == node_1
  assert new_edge.node2 == node_2
  assert new_edge.axis1 == 1
  assert new_edge.axis2 == 0
  assert new_edge.dimension == 2

  expected_tensor_1 = np.array([[1, 2], [4, 5], [7, 8]])
  expected_tensor_2 = np.array([[4, 5, 6, 7], [8, 9, 10, 11]])
  np.testing.assert_allclose(expected_tensor_1, node_1.get_tensor())
  np.testing.assert_allclose(expected_tensor_2, node_2.get_tensor())


def test_remove_trace_edge_dangling_edge_raises_value_error(single_node_edge):
  node = single_node_edge.node
  edge = node[0]
  edge.name = "e"
  with pytest.raises(ValueError, match="Attempted to remove dangling edge 'e"):
    _remove_trace_edge(edge, node)


def test_remove_trace_edge_non_trace_raises_value_error(double_node_edge):
  node1 = double_node_edge.node1
  node2 = double_node_edge.node2
  edge = tn.connect(node1[0], node2[0])
  edge.name = "e"
  with pytest.raises(ValueError, match="Edge 'e' is not a trace edge."):
    _remove_trace_edge(edge, node1)


def test_remove_edges_trace_raises_value_error(single_node_edge):
  node = single_node_edge.node
  edge = tn.connect(node[1], node[2])
  with pytest.raises(ValueError):
    _remove_edges(edge, node, node, node)  # pytype: disable=wrong-arg-types


def test_sparse_shape(backend):
  node = Node(tensor=np.random.rand(3, 4, 5), backend=backend)
  np.testing.assert_allclose(node.sparse_shape, (3, 4, 5))


def test_tensor_from_edge_order(backend):
  node = tn.Node(np.random.rand(2, 3, 4), backend=backend)
  order = [2, 0, 1]
  transp_tensor = node.tensor_from_edge_order([node[o] for o in order])
  np.testing.assert_allclose(transp_tensor.shape, [4, 2, 3])


def test_tensor_from_edge_order_raises(backend):
  node = tn.Node(np.random.rand(2, 3, 4), backend=backend)
  node2 = tn.Node(np.random.rand(2, 3, 4), backend=backend)
  with pytest.raises(ValueError):
    node.tensor_from_edge_order([node[1], node2[1], node[2]])


def test_copy(backend):
  node = tn.Node(np.random.rand(2, 2, 2, 2), backend=backend)
  node[3] ^ node[0]
  # should not raise
  copy = node.copy()
  np.testing.assert_allclose(copy.tensor, node.tensor)
