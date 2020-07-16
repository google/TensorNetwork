"""Tests for decorators."""
import pytest
import numpy as np
import functools
from tensornetwork.backends.abstract_backend import AbstractBackend
from tensornetwork.backends import backend_factory
from tensornetwork import backends
import tensornetwork

def jittest_init(backend):
  """
  Helper to initialize data for the other Jit tests.
  """
  backend_obj = backends.backend_factory.get_backend(backend)
  def fun(x, A, y):
    return backend_obj.multiply(x, backend_obj.multiply(A, y))
  x = backend_obj.randn((4,), seed=11)
  y = backend_obj.randn((4,), seed=11)
  A = backend_obj.randn((4, 4), seed=11)
  return (x, y, A, fun)


def test_jit(backend):
  """
  Tests that tn.jit gives the right answer.
  """
  x, y, A, fun = jittest_init(backend)
  fun_jit = tensornetwork.jit(fun, backend=backend)
  res1 = fun(x, A, y)
  res2 = fun_jit(x, A, y)
  np.testing.assert_allclose(res1, res2)


def test_jit_ampersand(backend):
  """
  Tests that tn.jit gives the right answer when used as a decorator.
  """
  x, y, A, fun = jittest_init(backend)
  @functools.partial(tensornetwork.jit, static_argnums=(3,), backend=backend)
  def fun_jit(x, A, y, dummy):
    _ = dummy
    return fun(x, A, y)
  res1 = fun(x, A, y)
  res2 = fun_jit(x, A, y, 2)
  np.testing.assert_allclose(res1, res2)


def test_jit_args(backend):
  """
  Tests that tn.jit gives the right answer when given extra arguments.
  """
  x, y, A, fun = jittest_init(backend)
  fun_jit = tensornetwork.jit(fun, backend=backend)
  res1 = fun(x, A, y)
  res2 = fun_jit(x, A, y)
  res3 = fun_jit(x, y=y, A=A)
  np.testing.assert_allclose(res1, res2)
  np.testing.assert_allclose(res1, res3)


def test_jit_backend_argnum_is_string(backend):
  """
  Tests that tn.jit gives the right answer when the backend is supplied
  via backend_argnum as a string.
  """
  x, y, A, fun = jittest_init(backend)

  @functools.partial(tensornetwork.jit, backend_argnum=3)
  def fun_jit(x, A, y, the_backend):
    _ = the_backend
    return fun(x, A, y)
  res1 = fun(x, A, y)
  res2 = fun_jit(x, A, y, backend)
  np.testing.assert_allclose(res1, res2)


def test_jit_backend_argnum_is_obj(backend):
  """
  Tests that tn.jit gives the right answer when the backend is supplied
  via backend_argnum as a backend object.
  """
  x, y, A, fun = jittest_init(backend)

  @functools.partial(tensornetwork.jit, backend_argnum=3)
  def fun_jit(x, A, y, the_backend):
    _ = the_backend
    return fun(x, A, y)
  res1 = fun(x, A, y)
  backend_obj = backends.backend_factory.get_backend(backend)
  res2 = fun_jit(x, A, y, backend_obj)
  np.testing.assert_allclose(res1, res2)


def test_jit_backend_argnum_invalid(backend):
  """
  Tests that tn.jit raises ValueError when backend_argnum points to something
  other than a backend.
  """
  x, y, A, fun = jittest_init(backend)

  with pytest.raises(ValueError):
    @functools.partial(tensornetwork.jit, backend_argnum=3)
    def fun_jit(x, A, y, the_backend):
      _ = the_backend
      return fun(x, A, y)
    _ = fun_jit(x, A, y, 99)


def test_jit_backend_and_backend_obj_raises_error(backend):
  """
  Tests that tn.jit raises ValueError when backend_argnum and backend
  are both specified.
  """
  x, y, A, fun = jittest_init(backend)

  with pytest.raises(ValueError):
    @functools.partial(tensornetwork.jit, backend_argnum=3, backend=backend)
    def fun_jit(x, A, y, the_backend):
      _ = the_backend
      return fun(x, A, y)
    _ = fun_jit(x, A, y, backend)
