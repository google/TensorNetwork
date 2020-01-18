import tensornetwork as tn
import numpy as np
import pytest

def test_throw_on_ambiguous_access():
  a = tn.Node(np.eye(3), axis_names=['a', 'b'])
  b = tn.Node(np.eye(3), axis_names=['a', 'b'])
  a['b'] ^ b['b']
  c = tn.contract_between(a, b, name='node')
  err_msg = "Axis name 'a' is ambiguous for node 'node'"
  with pytest.raises(ValueError, match=err_msg):
    c['a']

# contract_between
def test_basic_contract_between():
  a = tn.Node(np.eye(3), axis_names=['a', 'b'])
  b = tn.Node(np.eye(3), axis_names=['a', 'b'])
  a['b'] ^ b['a']
  c = a @ b
  assert(c.axis_names == ['a', 'b'])

def test_generic_names_contract_between():
  a = tn.Node(np.ones([2, 3, 4]))
  b = tn.Node(np.ones([2, 5, 4]))
  a[0] ^ b[0]
  a[2] ^ b[2]
  c = a @ b
  assert(c.axis_names == ['0', '1'])

def test_multi_edge_contract_between():
  a = tn.Node(np.ones([2, 3, 4]), axis_names=['a', 'b', 'c'])
  b = tn.Node(np.ones([2, 5, 4]), axis_names=['d', 'e', 'f'])
  a['a'] ^ b['d']
  a['c'] ^ b['f']
  c = tn.contract_between(a, b)
  assert(c.axis_names == ['b', 'e'])

def test_multi_edge_reverse_contract_between():
  a = tn.Node(np.ones([2, 3, 4]), axis_names=['a', 'b', 'c'])
  b = tn.Node(np.ones([2, 5, 4]), axis_names=['d', 'e', 'f'])
  a['a'] ^ b['d']
  a['c'] ^ b['f']
  c = tn.contract_between(b, a)
  assert(c.axis_names == ['e', 'b'])

def test_multi_edge_with_trace_contract_between():
  a = tn.Node(np.ones([2, 4, 4]), axis_names=['a', 'b', 'c'])
  b = tn.Node(np.ones([2, 5, 4]), axis_names=['d', 'e', 'f'])
  a['a'] ^ b['d']
  a['b'] ^ a['c']
  c = tn.contract_between(a, b)
  assert(c.axis_names == ['b', 'c', 'e', 'f'])
  d = tn.contract_between(c, c)
  assert(d.axis_names == ['e', 'f'])

def test_reorder_edges_contract_between():
  a = tn.Node(np.ones([2, 3, 4]), axis_names=['a', 'b', 'c'])
  b = tn.Node(np.ones([2, 5, 4]), axis_names=['d', 'e', 'f'])
  a['a'] ^ b['d']
  a['c'] ^ b['f']
  c = tn.contract_between(a, b, output_edge_order=[b['e'], a['b']])
  assert(c.axis_names == ['e', 'b'])

# contract
def test_basic_contract():
  a = tn.Node(np.ones([2, 4, 4]), axis_names=['a', 'b', 'c'])
  b = tn.Node(np.ones([2, 5, 4]), axis_names=['d', 'e', 'f'])
  a['a'] ^ b['d']
  c = tn.contract(a['a'])
  assert(c.axis_names == ['b', 'c', 'e', 'f'])

def test_generic_names_contract():
  a = tn.Node(np.ones([2, 4, 4]))
  b = tn.Node(np.ones([2, 5, 4]))
  a[0] ^ b[0]
  c = tn.contract(a[0])
  assert(c.axis_names == ['0', '1', '2', '3'])

def test_multi_contract():
  a = tn.Node(np.ones([2, 3, 4]), axis_names=['a', 'b', 'c'])
  b = tn.Node(np.ones([2, 3, 6]), axis_names=['d', 'e', 'f'])
  a['b'] ^ b['e']
  a['a'] ^ b['d']
  c = tn.contract(a['b'])
  assert(c.axis_names == ['a', 'c', 'd', 'f'])
  d = tn.contract(c['a'])
  assert(d.axis_names == ['c', 'f'])

# contract_trace
def test_contract_trace():
  a = tn.Node(np.ones([2, 3, 4, 3]), axis_names=['a', 'b', 'c', 'd'])
  a['b'] ^ a['d']
  b = tn.contract(a['b'])
  assert(b.axis_names == ['a', 'c'])

def test_generic_names_contract_trace():
  a = tn.Node(np.ones([2, 3, 2, 4]))
  a[0] ^ a[2]
  b = tn.contract(a[0])
  assert(b.axis_names == ['0', '1'])

# outer_product
def test_ambiguous_outer_product():
  a = tn.Node(np.ones([3]), axis_names=['a'])
  b = tn.Node(np.ones([4]), axis_names=['a'])
  c = tn.contract_between(a, b, allow_outer_product=True)
  assert(c.axis_names == ['a', 'a'])

def test_large_outer_product():
  a = tn.Node(np.ones([2, 3]), axis_names=['a', 'b'])
  b = tn.Node(np.ones([4, 5, 6]), axis_names=['c', 'd', 'e'])
  c = tn.contract_between(a, b, allow_outer_product=True)
  assert(c.axis_names == ['a', 'b', 'c', 'd', 'e'])
  d = tn.contract_between(b, a, allow_outer_product=True)
  assert(d.axis_names == ['c', 'd', 'e', 'a', 'b'])

def test_generic_names_outer_product():
  a = tn.Node(np.ones([3]))
  b = tn.Node(np.ones([4]))
  c = tn.outer_product(a, b)
  assert(c.axis_names == ['0', '1'])