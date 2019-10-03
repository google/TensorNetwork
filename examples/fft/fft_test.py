# python3
"""Tests for fft."""

import numpy as np
from examples.fft import fft
import tensornetwork as tn


def test_fft():
  n = 3
  initial_state = [complex(0)] * (1 << n)
  initial_state[1] = 1j
  initial_state[5] = -1
  initial_node = tn.Node(np.array(initial_state).reshape((2,) * n))

  fft_out = fft.add_fft([initial_node[k] for k in range(n)])
  result = tn.contractors.greedy(tn.reachable(fft_out[0].node1), fft_out)
  tn.flatten_edges(fft_out)
  actual = result.tensor
  expected = np.fft.fft(initial_state, norm="ortho")
  np.testing.assert_allclose(expected, actual)
