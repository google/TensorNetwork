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
"""TensorNetwork implementation of MPS image classfier.

The idea of using Matrix Product States for image classification was introduced
by Stoudenmire and Schwab in arXiv:1605.05775. This codes provides a
TensorNetwork / TensorFlow implementation of this idea. The implementation
differs from the original in various aspects with the most important being the
optimization part. In the original work as sweeping (DMRG inspired) algorithm
was used, while here we exploit the automatic differentation which is built-in
in machine learning libraries such as TensorFlow. Therefore we only need to
code the forward (prediction) part of the algorithm which is equivalent to
calculating the inner product between the data vector and the MPS vector.
More details can be found in the paper `TensorNetwork for Machine Learning` and
the implementation was inspired by jemisjoky/TorchMPS.

This file contains MatrixProductState and Environment classes which are
described in their definitions. The first defines the forward pass we need in
order to train and predict, while the latter is used as a utility to perform
the inner product calculations required.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensornetwork
from experiments.MPS_classifier import batchtensornetwork
from typing import Tuple, List, Optional


def random_initializer(d_phys: int, d_bond: int, std: float = 1e-3,
                       boundary: bool = False) -> np.ndarray:
  """Initializes MPS tensors randomly and close to identity matrices.

  Args:
    d_phys: Physical dimension of MPS.
    d_bond: Bond dimension of MPS.
    std: STD of normal distribution for random initialization.
    boundary: If True returns a tensor of shape (d_phys, d_bond).
      Otherwise returns a tensor of shape (d_phys, d_bond, d_bond).
    Note that d_phys given in this function does not have to be the actual
    MPS physical dimension (eg. it can also be n_labels to initialize there
    label MPS tensor).

  Returns:
    tensor: Random numpy array with shape described above.
  """
  if boundary:
    x = np.zeros((d_phys, d_bond))
    x[:, 0] = 1
  else:
    x = np.array(d_phys * [np.eye(d_bond)])
  x += np.random.normal(0.0, std, size=x.shape)
  return x


class Environment:
  """MatrixProductState environments.

  Perform the core calculation required for the inner product by building the
  relevant TensorNetwork. An environment consists of a boundary vector of shape
  (d_phys, d_bond) and the MPS matrices of shape (d_phys, d_bond, d_bond).
  Note that the MPS matrices are stored as a tensor of shape
  (n_sites, d_phys, d_bond, d_bond), namely with an additional "space" index
  for efficiency.  Upon contracting each environment with the corresponding
  data vector part, we contract the "space" index using
  `batchtensornetwork.pairwise_reduction` which performs a parallelized
  contraction of the MPS matrices which was found to be more efficient when
  implementing the automatic gradient optimization. We note that this type of
  contraction works because the input data corresponds to a product
  (non-entangled) state. Finally the result of the "space" index contraction
  is contracted with the boundary to give a tensor of shape (n_batch, d_bond).
  """

  def __init__(self, n_sites: int, d_phys: int,
               d_bond: int, std: float = 1e-3, dtype=tf.float32):
    self.n_sites, self.dtype = n_sites, dtype
    self.d_phys, self.d_bond = d_phys, d_bond

    v = random_initializer(d_phys, d_bond, std=std, boundary=True)
    self.vector = tf.Variable(v, dtype=dtype)
    m = random_initializer(d_phys * (n_sites - 1), d_bond, std=std)
    self.matrices = tf.Variable(
        m.reshape((n_sites - 1, d_phys, d_bond, d_bond)), dtype=dtype)

  def create_network(self, data: tf.Tensor, data0: tf.Tensor
                    ) -> Tuple[batchtensornetwork.BatchTensorNetwork,
                               List[tensornetwork.Node],
                               Tuple[tensornetwork.Node]]:
    """Creates TensorNetwork with MPS and data.
  
    Args:
      data: Tensor of input data of shape (n_batch, n_sites, d_phys).
      data0: Tensor of input data at the boundary of shape (n_batch, d_phys).
  
    Returns:
      net: TensorNetwork object containing the nodes.
      var_nodes: List of the two MPS nodes.
      data_nodes: Tuple of the two data nodes.
    """
    net = batchtensornetwork.BatchTensorNetwork()

    # Connect the bond edges of the MPS tensors
    var_nodes = [net.add_node(self.vector), net.add_node(self.matrices)]
    net.connect(var_nodes[0][1], var_nodes[1][2])

    # Connect the data nodes with the physical edges of the MPS tensors
    data_nodes = (net.add_node(data0), net.add_node(data))
    net.connect(data_nodes[0][1], var_nodes[0][0])
    net.connect(data_nodes[1][2], var_nodes[1][1])

    return net, var_nodes, data_nodes

  @staticmethod
  def contract_network(net: batchtensornetwork.BatchTensorNetwork,
                       var_nodes: List[tensornetwork.Node],
                       data_nodes: Tuple[tensornetwork.Node]) -> tf.Tensor:
    """Contracts TensorNetwork created in `create_network`."""
    batch_edges = tuple((x[0] for x in data_nodes))
    space_edge = data_nodes[1][1]

    # Contract data with the MPS tensors
    var_nodes[0] = net.contract_between(data_nodes[0], var_nodes[0])
    var_nodes[1] = net.batched_contract_between(data_nodes[1], var_nodes[1],
                                                space_edge, var_nodes[1][0])

    # Contract the artificial "space" index. This step is equivalent to
    # contracting the MPS over the bond dimensions.
    var_nodes[1] = batchtensornetwork.pairwise_reduction(net, var_nodes[1],
                                                         space_edge)
    # Contract the final bond edge with the boundary.
    var_nodes = net.batched_contract_between(var_nodes[0], var_nodes[1],
                                             batch_edges[0], batch_edges[1])
    return var_nodes.tensor

  def predict(self, data: tf.Tensor, data0: tf.Tensor) -> tf.Tensor:
    net, var_nodes, data_nodes = self.create_network(data, data0)
    return self.contract_network(net, var_nodes, data_nodes)


class MatrixProductState:
  """MPS classifier prediction graph.

  Contains the MPS tensors which are our variational parameters and
  methods that define the forward pass. These methods are used by `training.py`
  to fit data using automatic differentation and can also be used for
  predictions. Each MatrixProductState consists of a left and right environment
  which are connected by the label tensor to get the final prediction. These
  environments are defined in the `Environment` class.
  """

  def __init__(self,
               n_sites: int,
               n_labels: int,
               d_phys: int,
               d_bond: int,
               l_position: Optional[int] = None,
               std: float = 1e-3,
               dtype=tf.float32):
    self.dtype = dtype
    if l_position is None:
      l_position = n_sites // 2
    self.position = l_position

    l = random_initializer(n_labels, d_bond, std=std)
    self.labeled = tf.Variable(l, dtype=dtype)
    self.left_env = Environment(l_position, d_phys, d_bond, std=std,
                                dtype=dtype)
    self.right_env = Environment(n_sites - l_position - 1, d_phys, d_bond,
                                 std=std, dtype=dtype)

    self.tensors = [self.left_env.vector, self.left_env.matrices, self.labeled,
                    self.right_env.matrices, self.right_env.vector]

  def flx(self, data: tf.Tensor) -> tf.Tensor:
    """Calculates prediction given by contracting input data with MPS.
  
    This is equivalent to the "forward pass" of a neural network.
  
    Args:
      data: Tensor with input data of shape (n_batch, n_sites, d_phys).
  
    Returns:
      flx: Prediction (value of the function f^l(x)) with
        shape (n_batch, n_labels).
    """
    left = self.left_env.predict(data[:, 1:self.position], data[:, 0])
    right = self.right_env.predict(data[:, -2:self.position-1:-1], data[:, -1])
    return tf.einsum("bl,olr,br->bo", left, self.labeled, right)

  def loss(self, data: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor]:
    """Calculates loss in a batch of (data, labels).

    Args:
      data: Tensor with input data of shape (n_batch, n_sites, d_phys).
      labels: Tensor with the corresponding labels of shape (n_batch, n_labels).

    Returns:
      loss: Loss of the given batch.
      logits: flx prediction as returned from self.flx method.
    """
    logits = self.flx(data)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels,
                                                                    logits))
    return loss, logits
