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
from unittest import mock
import tensorflow as tf
# pylint: disable=g-import-not-at-top
from tensornetwork import network, nxwrapper


class TreePlotWrapperTest(tf.test.TestCase):

  def test_wrapping_mps(self):
    net = network.TensorNetwork()
    mps_nodes = [net.add_node(np.ones([2, 3]), name="W0")]
    for i in range(1, 5):
      mps_nodes.append(net.add_node(np.ones([3, 2, 3]), name="W{}".format(i)))
      net.connect(mps_nodes[i][0], mps_nodes[i - 1][-1])
    mps_nodes.append(net.add_node(np.ones([3, 2]), name="W{}".format(i)))
    net.connect(mps_nodes[-1][0], mps_nodes[-2][-1])
    plotter = nxwrapper.TreePlotWrapper(mps_nodes)
    self.assertEqual(len(plotter.levels), 1)
    self.assertEqual(plotter.ghost_counter, 6)
    self.assertListEqual(mps_nodes, plotter.levels[0])
    nodes_set = set(mps_nodes) | set(["ghost{}".format(i) for i in range(6)])
    self.assertSetEqual(nodes_set, plotter.node_set)

  def test_wrapping_bintree(self):
    net = network.TensorNetwork()
    tree_levels = []
    root = net.add_node(np.eye(4), name="R")
    leaves = [net.add_node(np.ones([4,]), name="W1{}".format(i))
              for i in range(2)]
    net.connect(root[0], leaves[0][0])
    net.connect(root[1], leaves[1][0])
    plotter = nxwrapper.TreePlotWrapper(root)
    self.assertEqual(plotter.ghost_counter, 0)
    # test BFS
    self.assertListEqual(plotter.levels, [[root], leaves])
    # test position assignment
    pos = {root: (1.5, 0), leaves[0]: (1, -1), leaves[1]: (2, -1)}
    self.assertDictEqual(pos, plotter.pos)

  def test_set_options(self):
    net = network.TensorNetwork()
    node = net.add_node(np.ones(5))
    plotter = nxwrapper.TreePlotWrapper(node)
    new_options = {"y_margin": 2.0, "sizes": 50, "top_to_bottom": False}
    full_options = {"x_margin": 1.0, "y_margin": 2.0,
                    "dangling_size": 0.8, "dangling_angle": 1.2,
                    "sizes": 50, "top_to_bottom": False}
    plotter.set_options(new_options)
    self.assertDictEqual(plotter.options, full_options)

  def test_colorsizemaps(self):
    net = network.TensorNetwork()
    node = net.add_node(np.ones(5))
    node2 = net.add_node(np.ones(5))
    net.connect(node[0], node2[0])
    plotter = nxwrapper.TreePlotWrapper(node)
    self.assertListEqual(plotter.colormap, ["red", "red"])
    self.assertListEqual(plotter.sizemap, [1000, 1000])

  @mock.patch("tensornetwork.nxwrapper.nx")
  def test_draw_tree(self, mock_nx):
    net = network.TensorNetwork()
    tree_levels = []
    root = net.add_node(np.eye(4), name="R")
    leaves = [net.add_node(np.ones([4,]), name="W1{}".format(i))
              for i in range(2)]
    net.connect(root[0], leaves[0][0])
    net.connect(root[1], leaves[1][0])
    nxwrapper.draw_tree(root)
    self.assertTrue(mock_nx.draw_networkx.called)


if __name__ == "__main__":
  tf.test.main()
