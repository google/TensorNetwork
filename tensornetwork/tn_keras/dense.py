# pylint: disable=no-name-in-module
import tensorflow as tf
from tensorflow.keras.layers import Layer  # type: ignore
from tensorflow.keras import activations
from tensorflow.keras import initializers
from typing import List, Optional, Text, Tuple
import tensornetwork as tn
import numpy as np


# pytype: disable=module-attr
@tf.keras.utils.register_keras_serializable(package='tensornetwork')# pylint: disable=no-member
# pytype: enable=module-attr
class DenseDecomp(Layer):
  """TN layer comparable to Dense that carries out matrix multiplication
  with 2 significantly smaller weight matrices instead of 1 large one.
  This layer is similar to performing a SVD on the weight matrix and dropping
  the lowest singular values.

  This layer can take an input shape of arbitrary dimension, with the first
  dimension expected to be a batch dimension. The weight matrix will be
  constructed from and applied to the last input dimension.

  Example:
    ::

      # as first layer in a sequential model:
      model = Sequential()
      model.add(
        DenseDecomp(512, 
                    decomp_size=128, 
                    activation='relu', 
                    input_shape=(1024,)))
      # now the model will take as input arrays of shape (*, 1024)
      # and output arrays of shape (*, 512).
      # After the first layer, you don't need to specify
      # the size of the input anymore:
      model.add(DenseDecomp(512, decomp_size=128, activation='relu'))

  Args:
    output_dim: Positive integer, dimensionality of the output space.
    decomp_size: Positive integer, size of the intermediate dimension. For
      TPU inference, it is recommended to use 128 or a small multiple of 128.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the two weight matrices.
    bias_initializer: Initializer for the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., output_dim)`.
  """

  def __init__(self,
               output_dim: int,
               decomp_size: int,
               use_bias: Optional[bool] = True,
               activation: Optional[Text] = None,
               kernel_initializer: Optional[Text] = 'glorot_uniform',
               bias_initializer: Optional[Text] = 'zeros',
               **kwargs) -> None:

    # Allow specification of input_dim instead of input_shape,
    # for compatability with Keras layers that support this
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super().__init__(**kwargs)

    self.output_dim = output_dim
    self.decomp_size = decomp_size

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

    self.a_var = self.add_weight(name='a',
                                 shape=(input_shape[-1], self.decomp_size),
                                 trainable=True,
                                 initializer=self.kernel_initializer)
    self.b_var = self.add_weight(name='b',
                                 shape=(self.decomp_size, self.output_dim),
                                 trainable=True,
                                 initializer=self.kernel_initializer)
    self.bias_var = self.add_weight(
        name='bias',
        shape=(self.output_dim,),
        trainable=True,
        initializer=self.bias_initializer) if self.use_bias else None

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor: # pylint: disable=unused-argument, arguments-differ


    def f(x: tf.Tensor, a_var: tf.Tensor, b_var: tf.Tensor, use_bias: bool,
          bias_var: tf.Tensor) -> tf.Tensor:
      a = tn.Node(a_var, backend="tensorflow")
      b = tn.Node(b_var, backend="tensorflow")
      x_node = tn.Node(x, backend="tensorflow")

      tn.connect(x_node[0], a[0])
      tn.connect(a[1], b[0])

      # The TN should now look like this
      #         |
      #         b
      #         |
      #         a
      #         |
      #         x

      c = a @ x_node
      result = (c @ b).tensor

      if use_bias:
        result += bias_var

      return result
    input_shape = list(inputs.shape)
    inputs = tf.reshape(inputs, (-1, input_shape[-1]))
    result = tf.vectorized_map(
        lambda vec: f(vec, self.a_var, self.b_var, self.use_bias, self.bias_var
                     ), inputs)
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

    # Include the DenseDecomp-specific arguments
    decomp_args = ['output_dim', 'decomp_size', 'use_bias']
    for arg in decomp_args:
      config[arg] = getattr(self, arg)

    # Serialize the activation
    config['activation'] = activations.serialize(getattr(self, 'activation'))

    # Serialize the initializers
    decomp_initializers = ['kernel_initializer', 'bias_initializer']
    for initializer_arg in decomp_initializers:
      config[initializer_arg] = initializers.serialize(
          getattr(self, initializer_arg))

    # Get base config
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
