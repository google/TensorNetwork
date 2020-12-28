import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations, initializers, regularizers
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils #pylint: disable=no-name-in-module
from typing import List, Tuple, Text, Optional, Union
import numpy as np
import tensornetwork as tn
import math

# pytype: disable=module-attr
@tf.keras.utils.register_keras_serializable(package='tensornetwork')
# pytype: enable=module-attr
class Conv2DMPO(Layer):
  """2D Convolutional Matrix Product Operator (MPO) TN layer.

  This layer recreates the functionality of a traditional convolutional
  layer, but stores the 'kernel' as a network of nodes forming an MPO.
  The bond dimension of the MPO can be adjusted to increase or decrease the
  number of parameters independently of the input and output dimensions.
  When the layer is called, the MPO is contracted into a traditional kernel
  and convolved with the layer input to produce a tensor of outputs.

  Example:
    ::

      # as first layer in a sequential model:
      model = Sequential()
      model.add(
        Conv2DMPO(256,
                  kernel_size=3,
                  num_nodes=4,
                  bond_dim=16,
                  activation='relu',
                  input_shape=(32, 32, 256)))
      # now the model will take as input tensors of shape (*, 32, 32, 256)
      # and output arrays of shape (*, 32, 32, 256).
      # After the first layer, you don't need to specify
      # the size of the input anymore:
      model.add(Conv2DMPO(256, 3, num_nodes=4, bond_dim=8, activation='relu'))

  Args:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    num_nodes: Positive integer, number of nodes in the MPO.
      Note input_shape[-1]**(1. / num_nodes) and filters**(1. / num_nodes)
      must both be round.
    bond_dim: Positive integer, size of the MPO bond dimension (between nodes).
      Lower bond dimension means more parameter compression.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution
      along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"`
    data_format: A string,
      one of `"channels_last"` or `"channels_first"`.
      The ordering of the dimensions in the inputs.
      `"channels_last"` corresponds to inputs with shape
      `(batch, height, width, channels)` while `"channels_first"`
      corresponds to inputs with shape
      `(batch, channels, height, width)`.
      It defaults to "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the node weight matrices.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer for the node weight matrices.
    bias_regularizer: Regularizer for the bias vector.

  Input shape:
    4D tensor with shape: `(batch_size, h, w, channels)`.

  Output shape:
    4D tensor with shape: `(batch_size, h_out, w_out, filters)`.
  """
  def __init__(self,
               filters: int,
               kernel_size: Union[int, Tuple[int, int]],
               num_nodes: int,
               bond_dim: int,
               strides: Union[int, Tuple[int, int]] = 1,
               padding: Text = "same",
               data_format: Optional[Text] = "channels_last",
               dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
               activation: Optional[Text] = None,
               use_bias: bool = True,
               kernel_initializer: Text = "glorot_uniform",
               bias_initializer: Text = "zeros",
               kernel_regularizer: Optional[Text] = None,
               bias_regularizer: Optional[Text] = None,
               **kwargs) -> None:
    if num_nodes < 2:
      raise ValueError('Need at least 2 nodes to create MPO')

    if padding not in ('same', 'valid'):
      raise ValueError('Padding must be "same" or "valid"')

    if data_format not in ['channels_first', 'channels_last']:
      raise ValueError('Invalid data_format string provided')

    super().__init__(**kwargs)

    self.nodes = []
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    self.num_nodes = num_nodes
    self.bond_dim = bond_dim
    self.strides = conv_utils.normalize_tuple(strides, 2, 'kernel_size')
    self.padding = padding
    self.data_format = data_format
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate,
                                                    2, 'dilation_rate')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

  def build(self, input_shape: List[int]) -> None:
    # Disable the attribute-defined-outside-init violations in this function
    # pylint: disable=attribute-defined-outside-init
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')

    def is_perfect_root(n, n_nodes):
      root = n**(1. / n_nodes)
      return round(root)**n_nodes == n

    channels = input_shape[channel_axis]

    # Ensure dividable dimensions
    assert is_perfect_root(channels, self.num_nodes), (
        f'Input dim incorrect. '
        f'{input_shape[-1]}**(1. / {self.num_nodes}) must be round.')

    assert is_perfect_root(self.filters, self.num_nodes), (
        f'Output dim incorrect. '
        f'{self.filters}**(1. / {self.num_nodes}) must be round.')

    super().build(input_shape)

    in_leg_dim = math.ceil(channels**(1. / self.num_nodes))
    out_leg_dim = math.ceil(self.filters**(1. / self.num_nodes))

    self.nodes.append(
        self.add_weight(name='end_node_first',
                        shape=(in_leg_dim, self.kernel_size[0],
                               self.bond_dim, out_leg_dim),
                        trainable=True,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer))
    for i in range(self.num_nodes - 2):
      self.nodes.append(
          self.add_weight(name=f'middle_node_{i}',
                          shape=(in_leg_dim, self.bond_dim, self.bond_dim,
                                 out_leg_dim),
                          trainable=True,
                          initializer=self.kernel_initializer,
                          regularizer=self.kernel_regularizer))
    self.nodes.append(
        self.add_weight(name='end_node_last',
                        shape=(in_leg_dim, self.bond_dim,
                               self.kernel_size[1], out_leg_dim),
                        trainable=True,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer))

    if self.use_bias:
      self.bias_var = self.add_weight(
          name='bias',
          shape=(self.filters,),
          trainable=True,
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer)
    else:
      self.use_bias = None

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:

    tn_nodes = [tn.Node(n, backend='tensorflow') for n in self.nodes]
    for i in range(len(tn_nodes) - 1):
      tn_nodes[i][2] ^ tn_nodes[i+1][1]
    input_edges = [n[0] for n in tn_nodes]
    output_edges = [n[3] for n in tn_nodes]
    edges = [tn_nodes[0][1], tn_nodes[-1][2]] + input_edges + output_edges

    contracted = tn.contractors.greedy(tn_nodes, edges)
    tn.flatten_edges(input_edges)
    tn.flatten_edges(output_edges)

    tf_df = 'NCHW' if self.data_format == 'channels_first' else 'NHWC'
    result = tf.nn.conv2d(inputs,
                          contracted.tensor,
                          self.strides,
                          self.padding.upper(),
                          data_format=tf_df,
                          dilations=self.dilation_rate)

    if self.use_bias:
      bias = tf.reshape(self.bias_var, (1, self.filters,))
      result += bias

    if self.activation is not None:
      result = self.activation(result)
    return result

  def compute_output_shape(self, input_shape: List[int]) -> Tuple[
      int, int, int, int]:
    if self.data_format == 'channels_first':
      space = input_shape[2:]
    else:
      space = input_shape[1:-1]
    new_space = []
    for i, _ in enumerate(space):
      new_dim = conv_utils.conv_output_length(
          space[i],
          self.kernel_size[i],
          padding=self.padding,
          stride=self.strides[i],
          dilation=self.dilation_rate[i])
      new_space.append(new_dim)
    if self.data_format == 'channels_first':
      return (input_shape[0], self.filters) + tuple(new_space)
    return (input_shape[0],) + tuple(new_space) + (self.filters,)

  def get_config(self) -> dict:
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'num_nodes': self.num_nodes,
        'bond_dim': self.bond_dim,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
    }
    base_config = super().get_config()
    config.update(base_config)
    return config
