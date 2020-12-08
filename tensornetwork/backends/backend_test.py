"""Tests for graphmode_tensornetwork."""
import builtins
import sys
import pytest
import numpy as np
from tensornetwork import connect, contract, Node
from tensornetwork.backends.abstract_backend import AbstractBackend
from tensornetwork.backends import backend_factory


def clean_tensornetwork_modules():
  for mod in list(sys.modules.keys()):
    if mod.startswith('tensornetwork'):
      sys.modules.pop(mod, None)


@pytest.fixture(autouse=True)
def clean_backend_import():
  #never do this outside testing
  clean_tensornetwork_modules()
  yield  # use as teardown
  clean_tensornetwork_modules()


@pytest.fixture
def no_backend_dependency(monkeypatch):
  import_orig = builtins.__import__

  # pylint: disable=redefined-builtin
  def mocked_import(name, globals, locals, fromlist, level):
    if name in ['torch', 'tensorflow', 'jax']:
      raise ImportError()
    return import_orig(name, globals, locals, fromlist, level)

  monkeypatch.setattr(builtins, '__import__', mocked_import)
  # Nuke the cache.
  backend_factory._INSTANTIATED_BACKENDS = dict()


@pytest.mark.usefixtures('no_backend_dependency')
def test_backend_pytorch_missing_cannot_initialize_backend():
  #pylint: disable=import-outside-toplevel
  with pytest.raises(ImportError):
    # pylint: disable=import-outside-toplevel
    from tensornetwork.backends.pytorch.pytorch_backend import PyTorchBackend
    PyTorchBackend()


@pytest.mark.usefixtures('no_backend_dependency')
def test_backend_tensorflow_missing_cannot_initialize_backend():
  #pylint: disable=import-outside-toplevel
  with pytest.raises(ImportError):
    # pylint: disable=import-outside-toplevel
    from tensornetwork.backends.tensorflow.tensorflow_backend \
      import TensorFlowBackend
    TensorFlowBackend()


@pytest.mark.usefixtures('no_backend_dependency')
def test_backend_jax_missing_cannot_initialize_backend():
  #pylint: disable=import-outside-toplevel
  with pytest.raises(ImportError):
    # pylint: disable=import-outside-toplevel
    from tensornetwork.backends.jax.jax_backend import JaxBackend
    JaxBackend()


@pytest.mark.usefixtures('no_backend_dependency')
def test_config_backend_missing_can_import_config():
  #not sure why config is imported here?
  #pylint: disable=import-outside-toplevel
  #pylint: disable=unused-variable
  import tensornetwork.config
  with pytest.raises(ImportError):
    #pylint: disable=import-outside-toplevel
    #pylint: disable=unused-variable
    import torch
  with pytest.raises(ImportError):
    #pylint: disable=import-outside-toplevel
    #pylint: disable=unused-variable
    import tensorflow as tf
  with pytest.raises(ImportError):
    #pylint: disable=import-outside-toplevel
    #pylint: disable=unused-variable
    import jax


@pytest.mark.usefixtures('no_backend_dependency')
def test_import_tensornetwork_without_backends():
  #pylint: disable=import-outside-toplevel
  #pylint: disable=unused-variable
  #pylint: disable=reimported
  import tensornetwork
  #pylint: disable=import-outside-toplevel
  import tensornetwork.backends.pytorch.pytorch_backend
  #pylint: disable=import-outside-toplevel
  import tensornetwork.backends.tensorflow.tensorflow_backend
  #pylint: disable=import-outside-toplevel
  import tensornetwork.backends.jax.jax_backend
  #pylint: disable=import-outside-toplevel
  import tensornetwork.backends.numpy.numpy_backend
  with pytest.raises(ImportError):
    #pylint: disable=import-outside-toplevel
    #pylint: disable=unused-variable
    import torch
  with pytest.raises(ImportError):
    #pylint: disable=unused-variable
    #pylint: disable=import-outside-toplevel
    import tensorflow as tf
  with pytest.raises(ImportError):
    #pylint: disable=unused-variable
    #pylint: disable=import-outside-toplevel
    import jax


@pytest.mark.usefixtures('no_backend_dependency')
def test_basic_numpy_network_without_backends():
  #pylint: disable=import-outside-toplevel
  #pylint: disable=reimported
  #pylint: disable=unused-variable
  import tensornetwork
  a = Node(np.ones((10,)), backend="numpy")
  b = Node(np.ones((10,)), backend="numpy")
  edge = connect(a[0], b[0])
  final_node = contract(edge)
  assert final_node.tensor == np.array(10.)
  with pytest.raises(ImportError):
    #pylint: disable=unused-variable
    #pylint: disable=import-outside-toplevel
    import torch
  with pytest.raises(ImportError):
    #pylint: disable=unused-variable
    #pylint: disable=import-outside-toplevel
    import tensorflow as tf
  with pytest.raises(ImportError):
    #pylint: disable=unused-variable
    #pylint: disable=import-outside-toplevel
    import jax


@pytest.mark.usefixtures('no_backend_dependency')
def test_basic_network_without_backends_raises_error():
  #pylint: disable=import-outside-toplevel
  #pylint: disable=reimported
  #pylint: disable=unused-variable
  import tensornetwork
  with pytest.raises(ImportError):
    Node(np.ones((2, 2)), backend="jax")
  with pytest.raises(ImportError):
    Node(np.ones((2, 2)), backend="tensorflow")
  with pytest.raises(ImportError):
    Node(np.ones((2, 2)), backend="pytorch")


def test_abstract_backend_name():
  backend = AbstractBackend()
  assert backend.name == "abstract backend"


def test_abstract_backend_tensordot_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.tensordot(np.ones((2, 2)), np.ones((2, 2)), axes=[[0], [0]])


def test_abstract_backend_reshape_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.reshape(np.ones((2, 2)), (4, 1))


def test_abstract_backend_transpose_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.transpose(np.ones((2, 2)), [0, 1])


def test_abstract_backend_slice_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.slice(np.ones((2, 2)), (0, 1), (1, 1))


def test_abstract_backend_svd_decompositon_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.svd(np.ones((2, 2)), 0)


def test_abstract_backend_qr_decompositon_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.qr(np.ones((2, 2)), 0)


def test_abstract_backend_cholesky_decompositon_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.cholesky(np.ones((2, 2)), 0)


def test_abstract_backend_rq_decompositon_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.rq(np.ones((2, 2)), 0)


def test_abstract_backend_shape_concat_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.shape_concat([np.ones((2, 2)), np.ones((2, 2))], 0)


def test_abstract_backend_shape_tensor_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.shape_tensor(np.ones((2, 2)))


def test_abstract_backend_shape_tuple_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.shape_tuple(np.ones((2, 2)))


def test_abstract_backend_shape_prod_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.shape_prod(np.ones((2, 2)))


def test_abstract_backend_sqrt_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.sqrt(np.ones((2, 2)))


def test_abstract_backend_convert_to_tensor_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.convert_to_tensor(np.ones((2, 2)))


def test_abstract_backend_trace_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.trace(np.ones((2, 2)))


def test_abstract_backend_outer_product_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.outer_product(np.ones((2, 2)), np.ones((2, 2)))


def test_abstract_backend_einsul_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.einsum("ii", np.ones((2, 2)), optimize=True)


def test_abstract_backend_norm_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.norm(np.ones((2, 2)))


def test_abstract_backend_eye_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.eye(2, dtype=np.float64)


def test_abstract_backend_ones_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.ones((2, 2), dtype=np.float64)


def test_abstract_backend_zeros_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.zeros((2, 2), dtype=np.float64)


def test_abstract_backend_randn_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.randn((2, 2))


def test_abstract_backend_random_uniforl_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.random_uniform((2, 2))


def test_abstract_backend_conj_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.conj(np.ones((2, 2)))


def test_abstract_backend_eigh_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.eigh(np.ones((2, 2)))


def test_abstract_backend_eigs_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.eigs(np.ones((2, 2)))


def test_abstract_backend_eigs_lanczos_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.eigsh_lanczos(lambda x: x, np.ones((2)))


def test_abstract_backend_gmres_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.gmres(lambda x: x, np.ones((2)))


def test_abstract_backend_addition_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.addition(np.ones((2, 2)), np.ones((2, 2)))


def test_abstract_backend_subtraction_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.subtraction(np.ones((2, 2)), np.ones((2, 2)))


def test_abstract_backend_multiply_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.multiply(np.ones((2, 2)), np.ones((2, 2)))


def test_abstract_backend_divide_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.divide(np.ones((2, 2)), np.ones((2, 2)))


def test_abstract_backend_index_update_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.index_update(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))


def test_abstract_backend_inv_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.inv(np.ones((2, 2)))


def test_abstract_backend_sin_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.sin(np.ones((2, 2)))


def test_abstract_backend_cos_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.cos(np.ones((2, 2)))


def test_abstract_backend_exp_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.exp(np.ones((2, 2)))


def test_abstract_backend_log_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.log(np.ones((2, 2)))


def test_abstract_backend_expm_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.expm(np.ones((2, 2)))


def test_abstract_backend_sparse_shape_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.sparse_shape(np.ones((2, 2)))


def test_abstract_backend_broadcast_right_multiplication_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.broadcast_right_multiplication(np.ones((2, 2)), np.ones((2, 2)))


def test_abstract_backend_broadcast_left_multiplication_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.broadcast_left_multiplication(np.ones((2, 2)), np.ones((2, 2)))


def test_backend_instantiation(backend):
  backend1 = backend_factory.get_backend(backend)
  backend2 = backend_factory.get_backend(backend)
  assert backend1 is backend2


def test_abstract_backend_diagonal_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.diagonal(np.ones((2, 2)))


def test_abstract_backend_diagflat_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.diagflat(np.ones((2, 2)))


def test_abstract_backend_sign_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.sign(np.ones((2, 2)))


def test_abstract_backend_abs_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.abs(np.ones((2, 2)))

def test_pivot_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.pivot(np.ones((2, 2)))

def test_power_not_implemented():
  backend = AbstractBackend()
  with pytest.raises(NotImplementedError):
    backend.power(np.array([1, 2]), np.array([1, 2]))
