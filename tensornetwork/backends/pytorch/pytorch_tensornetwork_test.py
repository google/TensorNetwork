"""Tests for graphmode_tensornetwork."""
import numpy as np
from tensornetwork import (connect, contract, contract_between,
                           flatten_edges_between, Node)
import torch


def test_basic_graphmode():
  a = Node(torch.ones(10), backend="pytorch")
  b = Node(torch.ones(10), backend="pytorch")
  e = connect(a[0], b[0])
  actual = contract(e).get_tensor()
  assert actual == 10.0


def test_gradient_decent():
  a = Node(
      torch.autograd.Variable(torch.ones(10), requires_grad=True),
      backend="pytorch")
  b = Node(torch.ones(10), backend="pytorch")
  e = connect(a[0], b[0])
  final_tensor = contract(e).get_tensor()
  opt = torch.optim.SGD([a.tensor], lr=0.001)
  opt.zero_grad()
  final_tensor.norm().backward()
  opt.step()
  np.testing.assert_allclose(final_tensor.data, 10)
  np.testing.assert_allclose(a.tensor.data, 0.999 * np.ones((10,)))
  assert final_tensor == 10


def test_dynamic_network_sizes():

  def f(x, n):
    x_slice = x[:n]
    n1 = Node(x_slice, backend="pytorch")
    n2 = Node(x_slice, backend="pytorch")
    e = connect(n1[0], n2[0])
    return contract(e).get_tensor()

  x = torch.ones(10)
  assert f(x, 2) == 2.
  assert f(x, 3) == 3.


def test_dynamic_network_sizes_contract_between():

  def f(x, n):
    x_slice = x[..., :n]
    n1 = Node(x_slice, backend="pytorch")
    n2 = Node(x_slice, backend="pytorch")
    connect(n1[0], n2[0])
    connect(n1[1], n2[1])
    connect(n1[2], n2[2])
    return contract_between(n1, n2).get_tensor()

  x = torch.ones((3, 4, 5))
  assert f(x, 2) == 24.
  assert f(x, 3) == 36.


def test_dynamic_network_sizes_flatten_standard():

  def f(x, n):
    x_slice = x[..., :n]
    n1 = Node(x_slice, backend="pytorch")
    n2 = Node(x_slice, backend="pytorch")
    connect(n1[0], n2[0])
    connect(n1[1], n2[1])
    connect(n1[2], n2[2])
    return contract(flatten_edges_between(n1, n2)).get_tensor()

  x = torch.ones((3, 4, 5))
  assert f(x, 2) == 24.
  assert f(x, 3) == 36.


def test_dynamic_network_sizes_flatten_trace():

  def f(x, n):
    x_slice = x[..., :n]
    n1 = Node(x_slice, backend="pytorch")
    connect(n1[0], n1[2])
    connect(n1[1], n1[3])
    return contract(flatten_edges_between(n1, n1)).get_tensor()

  x = torch.ones((3, 4, 3, 4, 5))
  np.testing.assert_allclose(f(x, 2), np.ones((2,)) * 12)
  np.testing.assert_allclose(f(x, 3), np.ones((3,)) * 12)
