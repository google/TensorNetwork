"""Tests for ncon."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tf.enable_v2_behavior()
from tensornetwork import ncon_interface


class NconTest(tf.test.TestCase):

  def test_ncon_sanity_check(self):
    result = ncon_interface.ncon(
        [tf.ones((2, 2)), tf.ones((2, 2))], [(-1, 0), (0, -2)])
    self.assertAllClose(result, tf.ones((2, 2)) * 2)


if __name__ == '__main__':
  tf.test.main()
