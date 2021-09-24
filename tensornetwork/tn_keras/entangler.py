# pylint: disable=no-name-in-module
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
@tf.keras.utils.register_keras_serializable(package='tensornetwork')# pylint: disable=no-member
# pytype: enable=module-attr
class DenseEntangler(Layer):
  """Entangler TN layer. Allows for very large hidden layers.

  This layer can take an input shape of arbitrary dimension, with the first
  dimension expected to be a batch dimension. The weight matrix will be
  constructed from and applied to the last input dimension.

  Example:
    ::  

      # as first layer in a sequential model:
      model = Sequential()
      model.add(
        DenseEntangler(16**3,
                        num_legs=3,
                        num_levels=3,
                        use_bias=True,
                        activation='relu',
                        input_shape=(16**3,)))
      # now the model will take as input arrays of shape (*, 4096)
      # and output arrays of shape (*, 4096).
      # After the first layer, you don't need to specify
      # the size of the input anymore:
      model.add(DenseEntangler(16**3, num_legs=3, num_levels=3, use_bias=True))

  Args:
    output_dim: Positive integer, dimensionality of the output space.
      Note: output_dim must be equal to the dimensionality of the input.
    num_legs: Positive integer, number of legs the state node has.
    num_levels: Positive integer, number of complete levels we want the
      entangler to have. A level consists of num_legs - 1 tensors, forming a
      complete layer of tensors connecting accross all legs.
      This is the only parameter that does not change input/output shape.
      It can be increased to increase the power of the layer, but inference
      time will also scale approximately linearly.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the weight matrices.
    bias_initializer: Initializer for the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., output_dim)`.
  """

  def __init__(self,
               output_dim: int,
               num_legs: int,
               num_levels: int,
               use_bias: Optional[bool] = True,
               activation: Optional[Text] = None,
               kernel_initializer: Optional[Text] = 'glorot_uniform',
               bias_initializer: Optional[Text] = 'zeros',
               **kwargs) -> None:

    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    assert (
        num_legs >=
        2), f'Need at least 2 legs to create Entangler but got {num_legs} legs'
    assert (
        num_levels >= 1
    ), f'Need at least 1 level to create Entangler but got {num_levels} levels'

    super().__init__(**kwargs)

    self.output_dim = output_dim
    self.num_legs = num_legs
    self.num_levels = num_levels
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

    def is_perfect_root(n, n_nodes):
      root = n**(1. / n_nodes)
      return round(root)**n_nodes == n

    super().build(input_shape)

    # Ensure the Entangler dimensions will work
    assert (
        is_perfect_root(input_shape[-1], self.num_legs)
    ), f'Input dim {input_shape[-1]}**(1. / {self.num_legs}) must be round.'

    assert (
        is_perfect_root(self.output_dim, self.num_legs)
    ), f'Output dim {self.output_dim}**(1. / {self.num_legs}) must be round.'

    self.leg_dim = round(input_shape[-1]**(1. / self.num_legs))
    self.out_leg_dim = round(self.output_dim**(1. / self.num_legs))
    self.num_nodes = self.num_levels * (self.num_legs - 1)

    for i in range(self.num_nodes):
      current_level = i // (self.num_legs - 1)
      a = b = c = d = min(self.leg_dim, self.out_leg_dim)
      if i == 0:
        a = self.leg_dim
      if i == self.num_nodes - 1:
        d = self.out_leg_dim
      if current_level == 0:
        b = self.leg_dim
      if current_level == self.num_levels - 1:
        c = self.out_leg_dim
      self.nodes.append(
          self.add_weight(name=f'node_{i}',
                          shape=(a, b, c, d),
                          trainable=True,
                          initializer=self.kernel_initializer))

    self.bias_var = self.add_weight(
        name='bias',
        shape=(self.output_dim,),
        trainable=True,
        initializer=self.bias_initializer) if self.use_bias else None

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:  # pylint: disable=unused-argument, arguments-differ

    def f(x: tf.Tensor, nodes: List[Node], num_nodes: int, num_legs: int,
          leg_dim: int, use_bias: bool, bias_var: tf.Tensor) -> tf.Tensor:

      l = [leg_dim] * num_legs
      input_reshaped = tf.reshape(x, tuple(l))

      x_node = tn.Node(input_reshaped, name='xnode', backend="tensorflow")
      edges = x_node.edges[:]  # force a copy
      for i in range(num_nodes):
        node = tn.Node(nodes[i], name=f'node_{i}', backend="tensorflow")
        tn.connect(edges[i % num_legs], node[0])
        tn.connect(edges[(i + 1) % num_legs], node[1])
        edges[i % num_legs] = node[2]
        edges[(i + 1) % num_legs] = node[3]
        x_node = tn.contract_between(x_node, node)

      # The TN will be connected in a "staircase" pattern, like this:
      #    |  |  |  |
      #    |  |  3333
      #    |  |  |  |
      #    |  2222  |
      #    |  |  |  |
      #    1111  |  |
      #    |  |  |  |
      #    xxxxxxxxxx

      result = tf.reshape(x_node.tensor, (self.output_dim,))
      if use_bias:
        result += bias_var

      return result

    input_shape = list(inputs.shape)
    inputs = tf.reshape(inputs, (-1, input_shape[-1]))
    result = tf.vectorized_map(
        lambda vec: f(vec, self.nodes, self.num_nodes, self.num_legs, self.
                      leg_dim, self.use_bias, self.bias_var), inputs)
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

    # Include the Entangler-specific arguments
    args = ['output_dim', 'num_legs', 'num_levels', 'use_bias']
    for arg in args:
      config[arg] = getattr(self, arg)

    # Serialize the activation
    config['activation'] = activations.serialize(getattr(self, 'activation'))

    # Serialize the initializers
    layer_initializers = ['kernel_initializer', 'bias_initializer']
    for initializer_arg in layer_initializers:
      config[initializer_arg] = initializers.serialize(
          getattr(self, initializer_arg))

    # Get base config
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
