# python3
"""Tests for fft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
# Prepare for TF 2.0 migration
tf.enable_v2_behavior()
from examples.fft import fft
import tensornetwork


class FftTest(tf.test.TestCase):

  def test_fft(self):
    n = 3
    net = tensornetwork.TensorNetwork()

    initial_state = [complex(0)] * (1 << n)
    initial_state[1] = 1j
    initial_state[5] = -1
    initial_node = net.add_node(np.array(initial_state).reshape((2,) * n))

    fft_out = fft.add_fft(net, [initial_node[k] for k in range(n)])
    net.check_correct()
    tensornetwork.contractors.naive(net)
    net.flatten_edges(fft_out)
    actual = net.get_final_node().tensor.numpy()
    expected = np.fft.fft(initial_state, norm="ortho")
    self.assertAllClose(expected, actual)


if __name__ == "__main__":
  tf.test.main()
