import pytest
import math
import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensornetwork.tn_keras.layers import Conv2DMPO

LAYER_NAME = 'conv_layer'


@pytest.fixture(params=[(100, 8, 8, 16)])
def dummy_data(request):
  np.random.seed(42)
  data = np.random.rand(*request.param)
  labels = np.concatenate((np.ones((50, 1)), np.zeros((50, 1))))
  return data, labels


@pytest.fixture()
def make_model(dummy_data):
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data
  model = Sequential()
  model.add(
      Conv2DMPO(filters=4,
                kernel_size=3,
                num_nodes=2,
                bond_dim=10,
                padding='same',
                input_shape=data.shape[1:],
                name=LAYER_NAME)
      )
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  return model


def test_train(dummy_data, make_model):
  # pylint: disable=redefined-outer-name
  model = make_model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  data, labels = dummy_data
  # Train the model for 10 epochs
  history = model.fit(data, labels, epochs=10, batch_size=32)

  # Check that loss decreases and accuracy increases
  assert history.history['loss'][0] > history.history['loss'][-1]
  assert history.history['accuracy'][0] < history.history['accuracy'][-1]


def test_weights_change(dummy_data, make_model):
  # pylint: disable=redefined-outer-name
  data, labels = dummy_data
  model = make_model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  before = model.get_weights()

  model.fit(data, labels, epochs=5, batch_size=32)

  after = model.get_weights()
  # Make sure every layer's weights changed
  for b, a in zip(before, after):
    assert (b != a).any()


def test_output_shape(dummy_data, make_model):
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data
  data = K.constant(data)
  model = make_model
  l = model.get_layer(LAYER_NAME)

  actual_output_shape = l(data).shape
  expected_output_shape = l.compute_output_shape(data.shape)
  np.testing.assert_equal(expected_output_shape, actual_output_shape)


def test_num_parameters(dummy_data, make_model):
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data
  model = make_model
  l = model.get_layer(LAYER_NAME)

  in_dim = math.ceil(data.shape[-1] ** (1. / l.num_nodes))
  out_dim = math.ceil(l.filters ** (1. / l.num_nodes))
  exp_num_parameters = ((l.num_nodes - 2) *
                        (l.bond_dim * 2 * in_dim * out_dim) +
                        (l.kernel_size[0] * out_dim * in_dim * l.bond_dim) +
                        (l.kernel_size[1] * out_dim * in_dim * l.bond_dim) +
                        (l.filters))
  np.testing.assert_equal(exp_num_parameters, l.count_params())


def test_config(make_model):
  # pylint: disable=redefined-outer-name
  model = make_model

  expected_num_parameters = model.layers[0].count_params()

  # Serialize model and use config to create new layer
  l = model.get_layer(LAYER_NAME)
  layer_config = l.get_config()
  new_model = Conv2DMPO.from_config(layer_config)

  # Build the layer so we can count params below
  new_model.build(layer_config['batch_input_shape'])

  np.testing.assert_equal(expected_num_parameters, new_model.count_params())
  assert layer_config == new_model.get_config()

def test_model_save(dummy_data, make_model, tmp_path):
  # pylint: disable=redefined-outer-name
  data, labels = dummy_data
  model = make_model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Train the model for 5 epochs
  model.fit(data, labels, epochs=5)

  for save_path in [tmp_path / 'test_model', tmp_path / 'test_model.h5']:
    # Save model to a SavedModel folder or h5 file, then load model
    print('save_path: ', save_path)
    model.save(save_path)
    loaded_model = load_model(save_path)

    # Clean up SavedModel folder
    if os.path.isdir(save_path):
      shutil.rmtree(save_path)

    # Clean up h5 file
    if os.path.exists(save_path):
      os.remove(save_path)

    # Compare model predictions and loaded_model predictions
    np.testing.assert_almost_equal(model.predict(data),
                                   loaded_model.predict(data))
