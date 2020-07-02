import numpy as np
from tensornetwork.block_sparse.charge_new import collapse, uncollapse
import pytest

def test_charge_collapse_uncollapse_1():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  assert a1b1.dtype == np.int16
  a1f, b1f = uncollapse(a1b1, [[a1.dtype], [b1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  assert a1b1.dtype == np.int32
  a1f, b1f = uncollapse(a1b1, [[a1.dtype], [b1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  assert a1b1.dtype == np.int32
  a1f, b1f = uncollapse(a1b1, [[a1.dtype], [b1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  c1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a1b1c1 = collapse([a1, b1, c1], [[a1.dtype], [b1.dtype], [c1.dtype]])
  assert a1b1c1.dtype == np.int32
  a1f, b1f, c1f = uncollapse(a1b1c1, [[a1.dtype], [b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(c1f, c1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  c1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a1b1c1 = collapse([a1, b1, c1], [[a1.dtype], [b1.dtype], [c1.dtype]])
  assert a1b1c1.dtype == np.int32
  a1f, b1f, c1f = uncollapse(a1b1c1, [[a1.dtype], [b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(c1f, c1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  c1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a1b1c1 = collapse([a1, b1, c1], [[a1.dtype], [b1.dtype], [c1.dtype]])
  assert a1b1c1.dtype == np.int32
  a1f, b1f, c1f = uncollapse(a1b1c1, [[a1.dtype], [b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(c1f, c1)

  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  c1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  a1b1c1 = collapse([a1, b1, c1], [[a1.dtype], [b1.dtype], [c1.dtype]])
  assert a1b1c1.dtype == np.int32
  a1f, b1f, c1f = uncollapse(a1b1c1, [[a1.dtype], [b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(c1f, c1)


def test_charge_collapse_uncollapse_2():
  np.random.seed(10)
  a = np.random.randint(-10, 10, 10000).astype(np.int8)
  b = np.random.randint(-10, 10, 10000).astype(np.int16)
  c = np.random.randint(-10, 10, 10000).astype(np.int8)

  r1 = np.left_shift(a.astype(np.int32), 24) + np.left_shift(
      b.astype(np.int32), 8) + c.astype(np.int32)
  r2 = np.left_shift(a.astype(np.int32), 24) + (
      np.left_shift(b.astype(np.int32), 8) + c.astype(np.int32))
  np.testing.assert_allclose(r1, r2)

  cf = np.bitwise_and(r1, 2**8 - 1).astype(np.int8)
  np.testing.assert_allclose(cf, c)
  r1 = np.right_shift(r1 - cf.astype(np.int32), 8)
  bf = np.bitwise_and(r1, 2**16 - 1).astype(np.int16)
  np.testing.assert_allclose(bf, b)
  af = np.right_shift(r1 - bf.astype(np.int32), 16).astype(np.int8)
  np.testing.assert_allclose(af, a)

  bc = collapse([b, c], [[b.dtype], [c.dtype]])
  abc = collapse([a, bc], [[a.dtype], [b.dtype, c.dtype]])
  abc2 = collapse([a, b, c], [[a.dtype], [b.dtype], [c.dtype]])
  np.testing.assert_allclose(abc, r2)
  np.testing.assert_allclose(abc2, r2)


def test_charge_collapse_uncollapse_3():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 10000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 10000).astype(np.int8)
  a2 = np.random.randint(-10, 10, 10000).astype(np.int8)
  b2 = np.random.randint(-10, 10, 10000).astype(np.int8)

  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  a2b2 = collapse([a2, b2], [[a2.dtype], [b2.dtype]])

  a1b1a2b2 = collapse([a1b1, a2b2],
                      [[a1.dtype, b1.dtype], [a2.dtype, b2.dtype]])
  a1b1_, a2b2_ = uncollapse(a1b1a2b2,
                            [[a1.dtype, b1.dtype], [a2.dtype, b2.dtype]])
  a1f, b1f = uncollapse(a1b1_, [[a1.dtype], [b1.dtype]])
  a2f, b2f = uncollapse(a2b2_, [[a2.dtype], [b2.dtype]])

  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(a2f, a2)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(b2f, b2)


def test_charge_collapse_uncollapse_4():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 10000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 10000).astype(np.int16)
  a2 = np.random.randint(-10, 10, 10000).astype(np.int8)

  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  a1b1a2 = collapse([a1b1, a2], [[a1.dtype, b1.dtype], [a2.dtype]])
  a1b1_, a2f = uncollapse(a1b1a2, [[a1.dtype, b1.dtype], [a2.dtype]])
  a1f, b1f = uncollapse(a1b1_, [[a1.dtype], [b1.dtype]])

  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(a2f, a2)
  np.testing.assert_allclose(b1f, b1)


def test_charge_collapse_uncollapse_5():
  np.random.seed(10)
  a = np.random.randint(-10, 10, 10000).astype(np.int8)
  b = np.random.randint(-10, 10, 10000).astype(np.int16)
  c = np.random.randint(-10, 10, 10000).astype(np.int8)

  bc = collapse([b, c], [[b.dtype], [c.dtype]])
  abc = collapse([a, bc], [[a.dtype], [b.dtype, c.dtype]])
  abc2 = collapse([a, b, c], [[a.dtype], [b.dtype], [c.dtype]])
  r1 = np.left_shift(a.astype(np.int32), 24) + np.left_shift(
      b.astype(np.int32), 8) + c.astype(np.int32)
  np.testing.assert_allclose(abc, r1)
  np.testing.assert_allclose(abc2, abc)

  af, bf, cf = uncollapse(abc, [[a.dtype], [b.dtype], [c.dtype]])
  np.testing.assert_allclose(af, a)
  np.testing.assert_allclose(bf, b)
  np.testing.assert_allclose(cf, c)
  af, bcf = uncollapse(abc, [[a.dtype], [b.dtype, c.dtype]])
  bf, cf = uncollapse(bcf, [[b.dtype], [c.dtype]])
  np.testing.assert_allclose(af, a)
  np.testing.assert_allclose(bf, b)
  np.testing.assert_allclose(cf, c)


def test_charge_collapse_uncollapse_6():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 1000000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 1000000).astype(np.int16)
  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  a2 = np.random.randint(-10, 10, 1000000).astype(np.int8)
  b2 = np.random.randint(-10, 10, 1000000).astype(np.int16)
  a2b2 = collapse([a2, b2], [[a2.dtype], [b2.dtype]])
  a1b1a2b2 = collapse([a1b1, a2b2],
                      [[a1.dtype, b1.dtype], [a2.dtype, b2.dtype]])
  a1b1_, a2b2_ = uncollapse(a1b1a2b2,
                            [[a1.dtype, b1.dtype], [a2.dtype, b2.dtype]])
  a1f, b1f = uncollapse(a1b1_, [[a1.dtype], [b1.dtype]])
  a2f, b2f = uncollapse(a2b2_, [[a2.dtype], [b2.dtype]])

  np.testing.assert_allclose(a1f, a1)
  np.testing.assert_allclose(a2f, a2)
  np.testing.assert_allclose(b1f, b1)
  np.testing.assert_allclose(b2f, b2)


def test_charge_collapse_uncollapse_7():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  a2 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b2 = np.random.randint(-10, 10, 100000).astype(np.int16)

  a1b1 = collapse([a1, b1], [[a1.dtype], [b1.dtype]])
  a2b2 = collapse([a2, b2], [[a2.dtype], [b2.dtype]])
  res = a1b1 + a2b2
  resa, resb = uncollapse(res, [[a1.dtype], [b1.dtype]])
  np.testing.assert_allclose(resa, a1 + a2)
  np.testing.assert_allclose(resb, b1 + b2)


def test_charge_collapse_uncollapse_8():
  np.random.seed(10)
  a1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b1 = np.random.randint(-10, 10, 100000).astype(np.int16)
  c1 = np.random.randint(-10, 10, 100000).astype(np.int8)
  a2 = np.random.randint(-10, 10, 100000).astype(np.int8)
  b2 = np.random.randint(-10, 10, 100000).astype(np.int16)
  c2 = np.random.randint(-10, 10, 100000).astype(np.int8)

  a1b1c1 = collapse([a1, b1, c1], [[a1.dtype], [b1.dtype], [c1.dtype]])
  a2b2c2 = collapse([a2, b2, c2], [[a2.dtype], [b2.dtype], [c2.dtype]])
  res = a1b1c1 + a2b2c2
  resa, resbc = uncollapse(res, [[a1.dtype], [b1.dtype, c1.dtype]])
  resb, resc = uncollapse(res, [[b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(resa, a1 + a2)
  np.testing.assert_allclose(resb, b1 + b2)
  np.testing.assert_allclose(resc, c1 + c2)
  resa2, resb2, resc2 = uncollapse(res, [[a1.dtype], [b1.dtype], [c1.dtype]])
  np.testing.assert_allclose(resa2, a1 + a2)
  np.testing.assert_allclose(resb2, b1 + b2)
  np.testing.assert_allclose(resc2, c1 + c2)
