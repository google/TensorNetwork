import numpy as np
import tensornetwork
from tensornetwork.contractors.opt_einsum_paths import optimal_path


def test_sanity_check(backend):
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
  final_node = optimal_path.optimal(net).get_final_node()
  assert final_node.shape == (13,)
