import numpy as np
import tensorflow as tf
import pytest
from collections import namedtuple
from tensornetwork.network_components import Node, Edge
import tensornetwork


SingleNodeEdgeTensor = namedtuple('SingleNodeEdgeTensor', 'node edge tensor')
DoubleNodeEdgeTensor = namedtuple('DoubleNodeEdgeTensor',
                                  'node1 node2 edge1 edge12 tensor')


@pytest.fixture(name='single_node_edge')
def fixture_single_node_edge(backend):
    net = tensornetwork.TensorNetwork(backend=backend)
    tensor = np.ones((1, 2, 2))
    tensor = net.backend.convert_to_tensor(tensor)
    node = Node(tensor=tensor, name="test_node",
                axis_names=["a", "b", "c"], network=net)
    edge = Edge(name="edge", node1=node, axis1=0)
    return SingleNodeEdgeTensor(node, edge, tensor)


@pytest.fixture(name='double_node_edge')
def fixture_double_node_edge(backend):
    net = tensornetwork.TensorNetwork(backend=backend)
    tensor = net.backend.convert_to_tensor(np.ones((1, 2, 2)))
    node1 = Node(tensor=tensor, name="test_node1",
                 axis_names=["a", "b", "c"], network=net)
    node2 = Node(tensor=tensor, name="test_node2",
                 axis_names=["a", "b", "c"], network=net)
    net.connect(node1["b"], node2["b"])
    edge1 = Edge(name="edge", node1=node1, axis1=0)
    edge12 = Edge(name="edge", node1=node1, axis1=1, node2=node2, axis2=1)
    return DoubleNodeEdgeTensor(node1, node2, edge1, edge12, tensor)


def test_node_initialize_numpy():
    net = tensornetwork.TensorNetwork(backend="numpy")
    tensor = np.ones((1, 2, 3))
    node = Node(tensor=tensor, name="test_node",
                axis_names=["a", "b", "c"], network=net)
    np.testing.assert_allclose(node.tensor, tensor)
    assert node.name == 'test_node'
    assert node.network == net
    assert len(node.edges) == 3
    assert isinstance(node.edges[0], Edge)
    assert node.axis_names == ["a", "b", "c"]
    assert node.signature == -1


def test_node_initialize_tensorflow():
    net = tensornetwork.TensorNetwork(backend="tensorflow")
    tensor = tf.ones((1, 2, 3))
    node = Node(tensor=tensor, name="test_node",
                axis_names=["a", "b", "c"], network=net)
    np.testing.assert_allclose(node.tensor, np.ones((1, 2, 3)))
    assert node.name == 'test_node'
    assert node.network == net
    assert len(node.edges) == 3
    assert isinstance(node.edges[0], Edge)
    assert node.axis_names == ["a", "b", "c"]
    assert node.signature == -1


def test_node_get_rank(single_node_edge):
    node = single_node_edge.node
    assert node.get_rank() == 3


def test_node_set_signature(single_node_edge):
    node = single_node_edge.node
    node.set_signature(2)
    assert node.signature == 2


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


def test_node_magic_getitem(single_node_edge):
    node = single_node_edge.node
    edge = single_node_edge.edge
    node.add_edge(edge, axis=0)
    assert node[0] == edge


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
        assert node < 0


def test_node_magic_matmul_raises_error_not_node(single_node_edge):
    node = single_node_edge.node
    with pytest.raises(TypeError):
        assert node @ 0


def test_node_magic_matmul_raises_error_different_network(single_node_edge):
    node = single_node_edge.node
    net = tensornetwork.TensorNetwork(backend=node.network.backend.name)
    tensor = net.backend.convert_to_tensor(np.zeros((1, 2, 3)))
    node2 = Node(tensor=tensor, name="test", axis_names=["A", "B", "C"],
                 network=net)
    with pytest.raises(ValueError):
        assert node @ node2


def test_node_magic_matmul(backend):
    net = tensornetwork.TensorNetwork(backend=backend)
    tensor1 = net.backend.convert_to_tensor(np.ones((2, 3, 4, 5)))
    tensor2 = net.backend.convert_to_tensor(2*np.ones((3, 5, 4, 2)))
    node1 = net.add_node(tensor1)
    node2 = net.add_node(tensor2)
    net.connect(node1[0], node2[3])
    net.connect(node2[1], node1[3])
    net.connect(node1[1], node2[0])
    actual = (node1 @ node2)
    expected = np.array([[60, 60, 60, 60], [60, 60, 60, 60],
                         [60, 60, 60, 60], [60, 60, 60, 60]])
    assert isinstance(actual, Node)
    np.testing.assert_allclose(actual.tensor, expected)


def test_edge_initialize_dangling(single_node_edge):
    node = single_node_edge.node
    edge = single_node_edge.edge
    assert edge.name == "edge"
    assert edge.node1 == node
    assert edge.axis1 == 0
    assert edge.node2 is None
    assert edge.axis2 is None
    assert edge.is_dangling() is True
    assert edge.signature == -1


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
    assert edge.signature == -1


def test_edge_initialize_raises_error_faulty_arguments(double_node_edge):
    node1 = double_node_edge.node1
    node2 = double_node_edge.node2
    with pytest.raises(ValueError):
        Edge(name="edge", node1=node1, node2=node2, axis1=0)
    with pytest.raises(ValueError):
        Edge(name="edge", node1=node1, axis1=0, axis2=0)


def test_edge_set_signature(double_node_edge):
    edge = double_node_edge.edge12
    edge.set_signature(2)
    assert edge.signature == 2


def test_edge_set_signature_raises_error_dangling(single_node_edge):
    edge = single_node_edge.edge
    with pytest.raises(ValueError):
        edge.set_signature(2)


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


def test_edge_magic_lt(double_node_edge):
    edge1 = double_node_edge.edge1
    edge2 = double_node_edge.edge12
    assert (edge1 < edge2) == (edge1.signature < edge2.signature)


def test_edge_magic_str(single_node_edge):
    edge = single_node_edge.edge
    assert(str(edge) == edge.name)
