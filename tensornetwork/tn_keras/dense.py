import tensorflow as tf
from tensorflow.keras.layers import Layer  # type: ignore
from tensorflow.keras import activations
from tensorflow.keras import initializers
from typing import List, Optional, Text, Tuple
import tensornetwork as tn
import numpy as np


class DenseDecomp(Layer):
  """TN layer comparable to Dense that carries out matrix multiplication
  with 2 significantly smaller weight matrices instead of 1 large one.
  This layer is similar to performing a SVD on the weight matrix and dropping
  the lowest singular values.

  Example:

  ```python
  # as first layer in a sequential model:
  model = Sequential()
  model.add(
    DenseDecomp(512, decomp_size=128, activation='relu', input_shape=(1024,)))
  # now the model will take as input arrays of shape (*, 1024)
  # and output arrays of shape (*, 512).
  # Note you can also specify input_dim=1024 instead of input_shape=(1024,),
  # as is sometimes done in other Keras layers like Dense.
  # After the first layer, you don't need to specify
  # the size of the input anymore:
  model.add(DenseDecomp(512, decomp_size=128, activation='relu'))
  ```

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
    2D tensor with shape: `(batch_size, input_shape[-1])`.

  Output shape:
    2D tensor with shape: `(batch_size, output_dim)`.
  """

  def __init__(self,
               output_dim: int,
               decomp_size: int,
               use_bias: Optional[bool] = True,
               activation: Optional[Text] = None,
               kernel_initializer: Optional[Text] = 'glorot_uniform',
               bias_initializer: Optional[Text] = 'zeros',
               **kwargs) -> None:
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(DenseDecomp, self).__init__(**kwargs)

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

    super(DenseDecomp, self).build(input_shape)

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

  def call(self, inputs: tf.Tensor) -> tf.Tensor:

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

    result = tf.vectorized_map(
        lambda vec: f(vec, self.a_var, self.b_var, self.use_bias, self.bias_var
                     ), inputs)
    if self.activation is not None:
      result = self.activation(result)
    return tf.reshape(result, (-1, self.output_dim))

  def compute_output_shape(self, input_shape: List[int]) -> Tuple[int, int]:
    return (input_shape[0], self.output_dim)

  def get_config(self) -> dict:
    """Returns the config of the layer.

    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    Returns:
      Python dictionary containing the configuration of the layer.
    """
    # Get base config
    config = super(DenseDecomp, self).get_config()

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

    return config
