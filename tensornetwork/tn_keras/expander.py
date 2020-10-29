import tensorflow as tf
from tensorflow.keras.layers import Layer  # type: ignore
from tensorflow.keras import activations
from tensorflow.keras import initializers
from typing import List, Optional, Text, Tuple
import tensornetwork as tn
from tensornetwork import Node
import numpy as np
import math


# pytype: disable=module-attr
@tf.keras.utils.register_keras_serializable(package='tensornetwork')
# pytype: enable=module-attr
class DenseExpander(Layer):
  """Expander TN layer. Greatly expands dimensionality of input.
  Used in conjunction with DenseEntangler to achieve very large hidden layers.

  This layer can take an input shape of arbitrary dimension, with the first
  dimension expected to be a batch dimension. The weight matrix will be
  constructed from and applied to the last input dimension.

  Example:
    ::

      # as first layer in a sequential model:
      model = Sequential()
      model.add(
        DenseExpander(exp_base=2
                      num_nodes=3,
                      use_bias=True,
                      activation='relu',
                      input_shape=(128,)))
      # now the model will take as input arrays of shape (*, 128)
      # and output arrays of shape (*, 1024).
      # After the first layer, you don't need to specify
      # the size of the input anymore:
      model.add(
        DenseExpander(exp_base=2, 
                      num_nodes=2, 
                      use_bias=True, 
                      activation='relu'))

  Args:
    exp_base: Positive integer, base of the dimensionality expansion term.
    num_nodes: Positive integer, number of nodes in expander.
      Note: the output dim will be input_shape[-1] * (exp_base**num_nodes)
      so increasing num_nodes will increase the output dim exponentially.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the two weight matrices.
    bias_initializer: Initializer for the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., input_shape[-1] *
                                              (exp_base**num_nodes))`.
  """

  def __init__(self,
               exp_base: int,
               num_nodes: int,
               use_bias: Optional[bool] = True,
               activation: Optional[Text] = None,
               kernel_initializer: Optional[Text] = 'glorot_uniform',
               bias_initializer: Optional[Text] = 'zeros',
               **kwargs) -> None:

    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super().__init__(**kwargs)

    self.exp_base = exp_base
    self.num_nodes = num_nodes
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

    super().build(input_shape)

    self.output_dim = input_shape[-1] * (self.exp_base**self.num_nodes)

    for i in range(self.num_nodes):
      self.nodes.append(
          self.add_weight(name=f'node_{i}',
                          shape=(input_shape[-1], self.exp_base,
                                 input_shape[-1]),
                          trainable=True,
                          initializer=self.kernel_initializer))

    self.bias_var = self.add_weight(
        name='bias',
        shape=(self.output_dim,),
        trainable=True,
        initializer=self.bias_initializer) if self.use_bias else None

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:  # pylint: disable=unused-argument

    def f(x: tf.Tensor, nodes: List[Node], num_nodes: int, use_bias: bool,
          bias_var: tf.Tensor) -> tf.Tensor:

      state_node = tn.Node(x, name='xnode', backend="tensorflow")
      operating_edge = state_node[0]

      # The TN will be connected like this:
      #     |    |   |   |
      #     |    |   33333
      #     |    |   |
      #     |    22222
      #     |    |
      #     11111
      #       |
      #    xxxxxxx

      for i in range(num_nodes):
        op = tn.Node(nodes[i], name=f'node_{i}', backend="tensorflow")
        tn.connect(operating_edge, op[0])
        operating_edge = op[2]
        state_node = tn.contract_between(state_node, op)

      result = tf.reshape(state_node.tensor, (-1,))

      if use_bias:
        result += bias_var

      return result

    input_shape = list(inputs.shape)
    inputs = tf.reshape(inputs, (-1, input_shape[-1]))
    result = tf.vectorized_map(
        lambda vec: f(vec, self.nodes, self.num_nodes, self.use_bias, self.
                      bias_var), inputs)
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

    # Include the Expander-specific arguments
    args = ['exp_base', 'num_nodes', 'use_bias']
    for arg in args:
      config[arg] = getattr(self, arg)

    # Serialize the activation
    config['activation'] = activations.serialize(getattr(self, 'activation'))

    # Serialize the initializers
    initializers_list = ['kernel_initializer', 'bias_initializer']
    for initializer_arg in initializers_list:
      config[initializer_arg] = initializers.serialize(
          getattr(self, initializer_arg))

    # Get base config
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
