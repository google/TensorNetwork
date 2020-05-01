import pytest
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensornetwork.tn_keras.dense import DenseDecomp


@pytest.fixture(params=[50, 100, 256])
def dummy_data(request):
  np.random.seed(42)
  # Generate dummy data for use in tests
  data = np.random.randint(10, size=(1000, request.param))
  labels = np.concatenate((np.ones((500, 1)), np.zeros((500, 1))), axis=0)
  return data, labels


def test_train(dummy_data):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, labels = dummy_data

  model = Sequential()
  model.add(
      DenseDecomp(512,
                  decomp_size=128,
                  use_bias=True,
                  activation='relu',
                  input_dim=data.shape[1]))
  model.add(DenseDecomp(256, decomp_size=64, activation='relu'))
  model.add(DenseDecomp(1, decomp_size=64, activation='sigmoid'))
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Train the model for 10 epochs
  history = model.fit(data, labels, epochs=10, batch_size=32)

  # Check that loss decreases and accuracy increases
  assert history.history['loss'][0] > history.history['loss'][-1]
  assert history.history['accuracy'][0] < history.history['accuracy'][-1]

def test_weights_change(dummy_data):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, labels = dummy_data

  model = Sequential()
  model.add(
      DenseDecomp(512,
                  decomp_size=128,
                  use_bias=True,
                  activation='relu',
                  input_dim=data.shape[1]))
  model.add(DenseDecomp(1, decomp_size=64, use_bias=True, activation='sigmoid'))
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  before = model.get_weights()

  model.fit(data, labels, epochs=5, batch_size=32)

  after = model.get_weights()
  # Make sure every layer's weights changed
  for i, _ in enumerate(before):
    assert (after[i] != before[i]).any()

def test_output_shape(dummy_data):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data
  data = K.constant(data)
  input_shape = data.shape

  dd = DenseDecomp(256,
                   decomp_size=128,
                   use_bias=False,
                   activation='relu',
                   input_dim=input_shape[1])

  actual_output_shape = dd(data).shape
  expected_output_shape = dd.compute_output_shape(input_shape)

  np.testing.assert_equal(expected_output_shape, actual_output_shape)


def test_num_parameters(dummy_data):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data
  output_dim = 256
  decomp_size = 128

  model = Sequential()
  model.add(
      DenseDecomp(output_dim,
                  decomp_size=decomp_size,
                  use_bias=True,
                  activation='relu',
                  input_dim=data.shape[1]))

  # num_params = a_params + b_params + bias_params
  expected_num_parameters = (data.shape[1] * decomp_size) + (
      decomp_size * output_dim) + output_dim

  np.testing.assert_equal(expected_num_parameters, model.count_params())
