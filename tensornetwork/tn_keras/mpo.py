import tensorflow as tf
from tensorflow.keras.layers import Layer  # type: ignore
from tensorflow.keras import activations
from tensorflow.keras import initializers
from typing import List, Optional, Text, Tuple
import tensornetwork as tn
from tensornetwork.network_components import Node
import numpy as np
import math


# pytype: disable=module-attr
@tf.keras.utils.register_keras_serializable(package='tensornetwork')
# pytype: enable=module-attr
class DenseMPO(Layer):
  """Matrix Product Operator (MPO) TN layer.

  This layer can take an input shape of arbitrary dimension, with the first
  dimension expected to be a batch dimension. The weight matrix will be
  constructed from and applied to the last input dimension.

  Example:
    ::

      # as first layer in a sequential model:
      model = Sequential()
      model.add(
        DenseMPO(1024, num_nodes=4, bond_dim=8, activation='relu',
        input_shape=(1024,)))
      # now the model will take as input arrays of shape (*, 1024)
      # and output arrays of shape (*, 1024).
      # After the first layer, you don't need to specify
      # the size of the input anymore:
      model.add(DenseMPO(1024, num_nodes=4, bond_dim=8, activation='relu'))

  Args:
    output_dim: Positive integer, dimensionality of the output space.
    num_nodes: Positive integer, number of nodes in the MPO.
      Note input_shape[-1]**(1. / num_nodes) and output_dim**(1. / num_nodes)
      must both be round.
    bond_dim: Positive integer, size of the intermediate dimension.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the node weight matrices.
    bias_initializer: Initializer for the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., output_dim)`.
  """

  def __init__(self,
               output_dim: int,
               num_nodes: int,
               bond_dim: int,
               use_bias: Optional[bool] = True,
               activation: Optional[Text] = None,
               kernel_initializer: Optional[Text] = 'glorot_uniform',
               bias_initializer: Optional[Text] = 'zeros',
               **kwargs) -> None:

    # Allow specification of input_dim instead of input_shape,
    # for compatability with Keras layers that support this
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    assert num_nodes > 2, 'Need at least 3 nodes to create MPO.'

    super().__init__(**kwargs)

    self.output_dim = output_dim
    self.num_nodes = num_nodes
    self.bond_dim = bond_dim
    self.nodes = []
    self.use_bias = use_bias
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

  def build(self, input_shape: List[int]) -> None:
    # Disable the attribute-defined-outside-init violations in this function
    # pylint: disable=attribute-defined-outside-init
    if input_shape[-1] is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # Try to convert n to an integer. tensorflow.compat.v1 uses a partially
    # integer compatible interface that does not implement the __pow__
    # function. __int__ is implemented, so calling this first is necessary.
    input_dim = int(input_shape[-1])

    def is_perfect_root(n, n_nodes):
      root = n**(1. / n_nodes)
      return round(root)**n_nodes == n

    # Ensure the MPO dimensions will work
    assert is_perfect_root(input_dim, self.num_nodes), \
      f'Input dim incorrect.\
      {input_dim}**(1. / {self.num_nodes}) must be round.'

    assert is_perfect_root(self.output_dim, self.num_nodes), \
      f'Output dim incorrect. \
      {self.output_dim}**(1. / {self.num_nodes}) must be round.'

    super().build(input_shape)

    self.in_leg_dim = math.ceil(input_dim**(1. / self.num_nodes))
    self.out_leg_dim = math.ceil(self.output_dim**(1. / self.num_nodes))

    self.nodes.append(
        self.add_weight(name='end_node_first',
                        shape=(self.in_leg_dim, self.bond_dim,
                               self.out_leg_dim),
                        trainable=True,
                        initializer=self.kernel_initializer))
    for i in range(self.num_nodes - 2):
      self.nodes.append(
          self.add_weight(name=f'middle_node_{i}',
                          shape=(self.in_leg_dim, self.bond_dim, self.bond_dim,
                                 self.out_leg_dim),
                          trainable=True,
                          initializer=self.kernel_initializer))
    self.nodes.append(
        self.add_weight(name='end_node_last',
                        shape=(self.in_leg_dim, self.bond_dim,
                               self.out_leg_dim),
                        trainable=True,
                        initializer=self.kernel_initializer))

    self.bias_var = self.add_weight(
        name='bias',
        shape=(self.output_dim,),
        trainable=True,
        initializer=self.bias_initializer) if self.use_bias else None

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:  # pylint: disable=unused-argument

    def f(x: tf.Tensor, nodes: List[Node], num_nodes: int, in_leg_dim: int,
          output_dim: int, use_bias: bool, bias_var: tf.Tensor) -> tf.Tensor:

      l = [in_leg_dim] * num_nodes
      input_reshaped = tf.reshape(x, tuple(l))
      x_node = tn.Node(input_reshaped, name='xnode', backend="tensorflow")

      tn_nodes = []
      for i, v in enumerate(nodes):
        tn_nodes.append(tn.Node(v, name=f'node_{i}', backend="tensorflow"))
        # Connect every node to input node
        x_node[i] ^ tn_nodes[i][0]

      # Connect all core nodes
      tn_nodes[0][1] ^ tn_nodes[1][1]
      for i, _ in enumerate(tn_nodes):
        if len(tn_nodes[i].shape) == 4:
          tn_nodes[i][2] ^ tn_nodes[i + 1][1]

      # The TN should now look like this
      #   |     |    |
      #   1 --- 2 --- ...
      #    \   /    /
      #      x

      # Contract TN using zipper algorithm
      temp = x_node @ tn_nodes[0]
      for i in range(1, len(tn_nodes)):
        temp = temp @ tn_nodes[i]

      result = tf.reshape(temp.tensor, (-1, output_dim))
      if use_bias:
        result += bias_var

      return result

    input_shape = list(inputs.shape)
    inputs = tf.reshape(inputs, (-1, input_shape[-1]))
    result = tf.vectorized_map(
        lambda vec: f(vec, self.nodes, self.num_nodes, self.in_leg_dim, self.
                      output_dim, self.use_bias, self.bias_var), inputs)
    if self.activation is not None:
      result = self.activation(result)
    result = tf.reshape(result, [-1] + input_shape[1:-1] + [self.output_dim,])
    return result

  def compute_output_shape(self, input_shape: List[int]) -> Tuple[int, int]:
    return tuple(input_shape[0:-1]) + (self.output_dim,)

  def get_config(self) -> dict:
    """Returns the config of the layer.

    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    Returns:
      Python dictionary containing the configuration of the layer.
    """
    config = {}

    # Include the MPO-specific arguments
    args = ['output_dim', 'num_nodes', 'bond_dim', 'use_bias']
    for arg in args:
      config[arg] = getattr(self, arg)

    # Serialize the activation
    config['activation'] = activations.serialize(getattr(self, 'activation'))

    # Serialize the initializers
    custom_initializers = ['kernel_initializer', 'bias_initializer']
    for initializer_arg in custom_initializers:
      config[initializer_arg] = initializers.serialize(
          getattr(self, initializer_arg))

    # Get base config
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
