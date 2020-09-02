import pytest
import numpy as np
import math
import os
import shutil
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model  # type: ignore
import tensorflow as tf
from tensornetwork.tn_keras.layers import DenseDecomp
from tensornetwork.tn_keras.layers import DenseMPO
from tensornetwork.tn_keras.layers import DenseCondenser
from tensornetwork.tn_keras.layers import DenseExpander
from tensornetwork.tn_keras.layers import DenseEntangler
from tensorflow.keras.layers import Dense  # type: ignore


@pytest.fixture(params=[512])
def dummy_data(request):
  np.random.seed(42)
  # Generate dummy data for use in tests
  data = np.random.randint(10, size=(1000, request.param))
  labels = np.concatenate((np.ones((500, 1)), np.zeros((500, 1))), axis=0)
  return data, labels


@pytest.fixture(params=[
    'DenseDecomp', 'DenseMPO', 'DenseCondenser', 'DenseExpander',
    'DenseEntangler', 'DenseEntanglerAsymmetric'
])
def make_model(dummy_data, request):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data

  if request.param == 'DenseMPO':
    model = Sequential()
    model.add(
        DenseMPO(data.shape[1],
                 num_nodes=int(math.log(int(data.shape[1]), 8)),
                 bond_dim=8,
                 use_bias=True,
                 activation='relu',
                 input_shape=(data.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))
  elif request.param == 'DenseDecomp':
    model = Sequential()
    model.add(
        DenseDecomp(512,
                    decomp_size=128,
                    use_bias=True,
                    activation='relu',
                    input_shape=(data.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))
  elif request.param == 'DenseCondenser':
    model = Sequential()
    model.add(
        DenseCondenser(exp_base=2,
                       num_nodes=3,
                       use_bias=True,
                       activation='relu',
                       input_shape=(data.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))
  elif request.param == 'DenseExpander':
    model = Sequential()
    model.add(
        DenseExpander(exp_base=2,
                      num_nodes=3,
                      use_bias=True,
                      activation='relu',
                      input_shape=(data.shape[-1],)))
    model.add(Dense(1, activation='sigmoid'))
  elif request.param == 'DenseEntangler':
    num_legs = 3
    leg_dim = round(data.shape[-1]**(1. / num_legs))
    assert leg_dim**num_legs == data.shape[-1]

    model = Sequential()
    model.add(
        DenseEntangler(leg_dim**num_legs,
                       num_legs=num_legs,
                       num_levels=3,
                       use_bias=True,
                       activation='relu',
                       input_shape=(data.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))
  elif request.param == 'DenseEntanglerAsymmetric':
    num_legs = 3
    leg_dim = round(data.shape[-1]**(1. / num_legs))
    assert leg_dim**num_legs == data.shape[-1]

    model = Sequential()
    model.add(
        DenseEntangler((leg_dim * 2)**num_legs,
                       num_legs=num_legs,
                       num_levels=3,
                       use_bias=True,
                       activation='relu',
                       input_shape=(data.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))

  return model


def test_train(dummy_data, make_model):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  tf.random.set_seed(0)
  data, labels = dummy_data
  model = make_model

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Train the model for 10 epochs
  history = model.fit(data, labels, epochs=10, batch_size=32)

  # Check that loss decreases and accuracy increases
  assert history.history['loss'][0] > history.history['loss'][-1]
  assert history.history['accuracy'][0] < history.history['accuracy'][-1]


def test_weights_change(dummy_data, make_model):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  tf.random.set_seed(0)
  data, labels = dummy_data
  model = make_model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  before = model.get_weights()

  model.fit(data, labels, epochs=5, batch_size=32)

  after = model.get_weights()
  # Make sure every layer's weights changed
  for i, _ in enumerate(before):
    assert (after[i] != before[i]).any()


def test_output_shape(dummy_data, make_model):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data
  data = K.constant(data)
  input_shape = data.shape

  model = make_model

  actual_output_shape = model(data).shape
  expected_output_shape = model.compute_output_shape(input_shape)

  np.testing.assert_equal(expected_output_shape, actual_output_shape)


@pytest.fixture(params=[(100, 10, 10, 512), (100, 512), (20, 10, 512)])
def high_dim_data(request):
  np.random.seed(42)
  # Generate dummy data for use in tests
  data = np.random.randint(10, size=request.param)
  return data


@pytest.fixture(params=[
    'DenseDecomp', 'DenseMPO', 'DenseCondenser', 'DenseExpander',
    'DenseEntangler'
])
def make_high_dim_model(high_dim_data, request):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data = high_dim_data

  if request.param == 'DenseMPO':
    model = Sequential()
    model.add(
        DenseMPO(data.shape[-1],
                 num_nodes=int(math.log(int(data.shape[-1]), 8)),
                 bond_dim=8,
                 use_bias=True,
                 activation='relu',
                 input_shape=(data.shape[-1],)))
  elif request.param == 'DenseDecomp':
    model = Sequential()
    model.add(
        DenseDecomp(512,
                    decomp_size=128,
                    use_bias=True,
                    activation='relu',
                    input_shape=(data.shape[-1],)))
  elif request.param == 'DenseCondenser':
    model = Sequential()
    model.add(
        DenseCondenser(exp_base=2,
                       num_nodes=3,
                       use_bias=True,
                       activation='relu',
                       input_shape=(data.shape[-1],)))
  elif request.param == 'DenseExpander':
    model = Sequential()
    model.add(
        DenseExpander(exp_base=2,
                      num_nodes=3,
                      use_bias=True,
                      activation='relu',
                      input_shape=(data.shape[-1],)))
  elif request.param == 'DenseEntangler':
    num_legs = 3
    leg_dim = round(data.shape[-1]**(1. / num_legs))
    assert leg_dim**num_legs == data.shape[-1]

    model = Sequential()
    model.add(
        DenseEntangler(leg_dim**num_legs,
                       num_legs=num_legs,
                       num_levels=3,
                       use_bias=True,
                       activation='relu',
                       input_shape=(data.shape[-1],)))

  return data, model


def test_higher_dim_input_output_shape(make_high_dim_model):
  # pylint: disable=redefined-outer-name
  data, model = make_high_dim_model

  actual_output_shape = model(data).shape
  expected_output_shape = model.compute_output_shape(data.shape)

  np.testing.assert_equal(expected_output_shape, actual_output_shape)


def test_decomp_num_parameters(dummy_data):
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
                  input_shape=(data.shape[1],)))

  # num_params = a_params + b_params + bias_params
  expected_num_parameters = (data.shape[1] * decomp_size) + (
      decomp_size * output_dim) + output_dim

  np.testing.assert_equal(expected_num_parameters, model.count_params())


def test_mpo_num_parameters(dummy_data):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data
  output_dim = data.shape[1]
  num_nodes = int(math.log(data.shape[1], 8))
  bond_dim = 8

  model = Sequential()
  model.add(
      DenseMPO(output_dim,
               num_nodes=num_nodes,
               bond_dim=bond_dim,
               use_bias=True,
               activation='relu',
               input_shape=(data.shape[1],)))

  in_leg_dim = math.ceil(data.shape[1]**(1. / num_nodes))
  out_leg_dim = math.ceil(output_dim**(1. / num_nodes))

  # num_params = num_edge_node_params + num_middle_node_params + bias_params
  expected_num_parameters = (2 * in_leg_dim * bond_dim * out_leg_dim) + (
      (num_nodes - 2) * in_leg_dim * bond_dim * bond_dim *
      out_leg_dim) + output_dim

  np.testing.assert_equal(expected_num_parameters, model.count_params())


def test_condenser_num_parameters(dummy_data):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data
  exp_base = 2
  num_nodes = 3
  model = Sequential()
  model.add(
      DenseCondenser(exp_base=exp_base,
                     num_nodes=num_nodes,
                     use_bias=True,
                     activation='relu',
                     input_shape=(data.shape[1],)))

  output_dim = data.shape[-1] // (exp_base**num_nodes)

  # num_params = (num_nodes * num_node_params) + num_bias_params
  expected_num_parameters = (num_nodes * output_dim * output_dim *
                             exp_base) + output_dim

  np.testing.assert_equal(expected_num_parameters, model.count_params())


def test_expander_num_parameters(dummy_data):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data
  exp_base = 2
  num_nodes = 3
  model = Sequential()
  model.add(
      DenseExpander(exp_base=exp_base,
                    num_nodes=num_nodes,
                    use_bias=True,
                    activation='relu',
                    input_shape=(data.shape[-1],)))

  output_dim = data.shape[-1] * (exp_base**num_nodes)

  # num_params = (num_nodes * num_node_params) + num_bias_params
  expected_num_parameters = (num_nodes * data.shape[-1] * data.shape[-1] *
                             exp_base) + output_dim

  np.testing.assert_equal(expected_num_parameters, model.count_params())


def test_entangler_num_parameters(dummy_data):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, _ = dummy_data

  num_legs = 3
  num_levels = 3
  leg_dim = round(data.shape[-1]**(1. / num_legs))
  assert leg_dim**num_legs == data.shape[-1]

  model = Sequential()
  model.add(
      DenseEntangler(leg_dim**num_legs,
                     num_legs=num_legs,
                     num_levels=num_levels,
                     use_bias=True,
                     activation='relu',
                     input_shape=(data.shape[1],)))

  # num_params = entangler_node_params + bias_params
  expected_num_parameters = num_levels * (num_legs - 1) * (leg_dim**4) + (
      leg_dim**num_legs)

  np.testing.assert_equal(expected_num_parameters, model.count_params())


@pytest.mark.parametrize('num_levels', list(range(1, 4)))
@pytest.mark.parametrize('num_legs', list(range(2, 6)))
@pytest.mark.parametrize('leg_dims', [(4, 8), (8, 4)])
def test_entangler_asymmetric_num_parameters_output_shape(num_legs,
                                                          num_levels,
                                                          leg_dims):
  leg_dim, out_leg_dim = leg_dims
  data_shape = (leg_dim ** num_legs,)
  model = Sequential()
  model.add(
      DenseEntangler(out_leg_dim**num_legs,
                     num_legs=num_legs,
                     num_levels=num_levels,
                     use_bias=True,
                     activation='relu',
                     input_shape=data_shape))


  primary = leg_dim
  secondary = out_leg_dim
  if leg_dim > out_leg_dim:
    primary, secondary = secondary, primary

  expected_num_parameters = (num_levels - 1) * (num_legs - 1) * (primary**4) + (
      num_legs - 2) * primary**3 * secondary + primary**2 * secondary**2 + (
          out_leg_dim**num_legs)


  np.testing.assert_equal(expected_num_parameters, model.count_params())
  data = np.random.randint(10, size=(10, data_shape[0]))
  out = model(data)
  np.testing.assert_equal(out.shape, (data.shape[0], out_leg_dim**num_legs))


def test_config(make_model):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  model = make_model

  expected_num_parameters = model.layers[0].count_params()

  # Serialize model and use config to create new layer
  model_config = model.get_config()
  layer_config = model_config['layers'][1]['config']
  if 'mpo' in model.layers[0].name:
    new_model = DenseMPO.from_config(layer_config)
  elif 'decomp' in model.layers[0].name:
    new_model = DenseDecomp.from_config(layer_config)
  elif 'condenser' in model.layers[0].name:
    new_model = DenseCondenser.from_config(layer_config)
  elif 'expander' in model.layers[0].name:
    new_model = DenseExpander.from_config(layer_config)
  elif 'entangler' in model.layers[0].name:
    new_model = DenseEntangler.from_config(layer_config)

  # Build the layer so we can count params below
  new_model.build(layer_config['batch_input_shape'])

  # Check that original layer had same num params as layer built from config
  np.testing.assert_equal(expected_num_parameters, new_model.count_params())


def test_model_save(dummy_data, make_model, tmp_path):
  # Disable the redefined-outer-name violation in this function
  # pylint: disable=redefined-outer-name
  data, labels = dummy_data
  model = make_model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Train the model for 5 epochs
  model.fit(data, labels, epochs=5, batch_size=32)

  for save_path in ['test_model', 'test_model.h5']:
    # Save model to a SavedModel folder or h5 file, then load model
    save_path = tmp_path / save_path
    model.save(save_path)
    loaded_model = load_model(save_path)

    # Clean up SavedModel folder
    if os.path.isdir(save_path):
      shutil.rmtree(save_path)

    # Clean up h5 file
    if os.path.exists(save_path):
      os.remove(save_path)

    # Compare model predictions and loaded_model predictions
    np.testing.assert_equal(model.predict(data), loaded_model.predict(data))
