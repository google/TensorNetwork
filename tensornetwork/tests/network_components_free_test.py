import numpy as np
import tensorflow as tf
import pytest
from collections import namedtuple
import h5py
from tensornetwork.network_components import Node, CopyNode, Edge
import tensornetwork as tn

string_type = h5py.special_dtype(vlen=str)

SingleNodeEdgeTensor = namedtuple('SingleNodeEdgeTensor', 'node edge tensor')
DoubleNodeEdgeTensor = namedtuple('DoubleNodeEdgeTensor',
                                  'node1 node2 edge1 edge12 tensor')


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
  assert node.signature == -1


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
    assert node < 0


def test_node_magic_matmul_raises_error_not_node(single_node_edge):
  node = single_node_edge.node
  with pytest.raises(TypeError):
    assert node @ 0


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


def test_node_save_structure(tmp_path, single_node_edge):
  node = single_node_edge.node
  with h5py.File(tmp_path / 'nodes', 'w') as node_file:
    node_group = node_file.create_group('test_node')
    node._save_node(node_group)
    assert set(list(node_file.keys())) == {"test_node"}
    assert set(list(node_file['test_node'])) == {
        "tensor", "signature", 'backend', 'name', 'edges', 'shape',
        'axis_names', "type"
    }


def test_node_save_data(tmp_path, single_node_edge):
  node = single_node_edge.node
  with h5py.File(tmp_path / 'nodes', 'w') as node_file:
    node_group = node_file.create_group('test_node')
    node._save_node(node_group)
    np.testing.assert_allclose(node_file['test_node/tensor'][()], node.tensor)
    assert node_file['test_node/signature'][()] == node.signature
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
    node_group.create_dataset('signature', data=node.signature)
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

    loaded_node = Node._load_node(net=None, node_data=node_file["node_data/"])
    assert loaded_node.name == node.name
    assert loaded_node.signature == node.signature
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
        "signature", 'name', 'edges', 'backend', 'shape', 'axis_names',
        'copy_node_dtype', "type"
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
    assert node_file['copier/signature'][()] == node.signature
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
    node_group.create_dataset('signature', data=node.signature)
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

    loaded_node = CopyNode._load_node(
        net=None, node_data=node_file["node_data/"])
    assert loaded_node.name == node.name
    assert loaded_node.signature == node.signature
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
  assert str(edge) == edge.name


def test_edge_node_save_structure(tmp_path, double_node_edge):
  edge12 = double_node_edge.edge12
  with h5py.File(tmp_path / 'edges', 'w') as edge_file:
    edge_group = edge_file.create_group('edge')
    edge12._save_edge(edge_group)
    assert set(list(edge_group.keys())) == {
        "axis1", "node1", "axis2", "node2", "name", "signature"
    }


def test_edge_node_save_data(tmp_path, double_node_edge):
  edge = double_node_edge.edge12
  with h5py.File(tmp_path / 'edges', 'w') as edge_file:
    edge_group = edge_file.create_group('edge')
    edge._save_edge(edge_group)
    assert edge_file['edge/signature'][()] == edge.signature
    assert edge_file['edge/name'][()] == edge.name
    assert edge_file['edge/node1'][()] == edge.node1.name
    assert edge_file['edge/node2'][()] == edge.node2.name
    assert edge_file['edge/axis1'][()] == edge.axis1
    assert edge_file['edge/axis2'][()] == edge.axis2


def test_edge_load(backend, tmp_path, double_node_edge):
  edge = double_node_edge.edge12

  with h5py.File(tmp_path / 'edge', 'w') as edge_file:
    edge_group = edge_file.create_group('edge_data')
    edge_group.create_dataset('signature', data=edge.signature)
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
    assert loaded_edge.signature == edge.signature
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
