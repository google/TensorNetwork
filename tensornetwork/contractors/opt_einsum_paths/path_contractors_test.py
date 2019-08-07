import numpy as np
import pytest
import tensornetwork
from tensornetwork.contractors.opt_einsum_paths import path_contractors


@pytest.fixture(name="path_algorithm",
                params=["optimal", "branch", "greedy", "auto"])
def path_algorithm_fixture(request):
  return getattr(path_contractors, request.param)


def test_sanity_check(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.eye(2))
  b = net.add_node(np.ones((2, 7, 11)))
  c = net.add_node(np.ones((7, 11, 13, 2)))
  d = net.add_node(np.eye(13))
  # pylint: disable=pointless-statement
  a[0] ^ b[0]
  b[1] ^ c[0]
  b[2] ^ c[1]
  c[2] ^ d[1]
  c[3] ^ a[1]
  final_node = path_algorithm(net).get_final_node()
  assert final_node.shape == (13,)


def test_trace_edge(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2, 2, 2)))
  b = net.add_node(np.ones((2, 2, 2)))
  c = net.add_node(np.ones((2, 2, 2)))
  # pylint: disable=pointless-statement
  a[0] ^ a[1]
  a[2] ^ b[0]
  a[3] ^ c[0]
  b[1] ^ c[1]
  b[2] ^ c[2]
  node = path_algorithm(net).get_final_node()
  np.testing.assert_allclose(node.tensor, np.ones(2) * 32.0)


def test_disconnected_network(backend, path_algorithm):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.array([2, 2]))
  b = net.add_node(np.array([2, 2]))
  c = net.add_node(np.array([2, 2]))
  d = net.add_node(np.array([2, 2]))
  # pylint: disable=pointless-statement
  a[0] ^ b[0]
  c[0] ^ d[0]
  with pytest.raises(ValueError):
    net = path_algorithm(net)


def test_auto_single_node(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones((2, 2, 2)))
  # pylint: disable=pointless-statement
  a[0] ^ a[1]
  node = path_contractors.auto(net).get_final_node()
  np.testing.assert_allclose(node.tensor, np.ones(2) * 2.0)


def test_custom_sanity_check(backend):
  net = tensornetwork.TensorNetwork(backend=backend)
  a = net.add_node(np.ones(2))
  b = net.add_node(np.ones((2, 5)))
  # pylint: disable=pointless-statement
  a[0] ^ b[0]

  class PathOptimizer:

    def __call__(self, inputs, output, size_dict, memory_limit=None):
      return [(0, 1)]

  optimizer = PathOptimizer()
  final_node = path_contractors.custom(net, optimizer).get_final_node()
  np.testing.assert_allclose(final_node.tensor, np.ones(5) * 2.0)
