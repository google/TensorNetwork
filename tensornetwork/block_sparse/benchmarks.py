import tensornetwork as tn
import numpy as np
import time
from tensornetwork.block_sparse.block_tensor import BlockSparseTensor, tensordot
from tensornetwork.block_sparse.index import Index
from tensornetwork.block_sparse.charge import U1Charge


def benchmark_1_U1():
  R = 6
  charges = [
      U1Charge(
          np.asarray([-2, -1, -1, 0, -1, 0, 0, 1, -1, 0, 0, 1, 0, 1, 1, 2],
                     dtype=np.int16)) for n in range(R)
  ]

  indsA = np.random.choice(np.arange(R), R // 2, replace=False)
  indsB = np.random.choice(np.arange(R), R // 2, replace=False)

  flowsA = np.asarray([False] * R)
  flowsB = np.asarray([False] * R)

  flowsB[indsB] = True
  A = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsA[n], name='a{}'.format(n)) for n in range(R)
  ])
  B = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsB[n], name='b{}'.format(n)) for n in range(R)
  ])

  final_order = np.arange(R)
  np.random.shuffle(final_order)
  t1 = time.time()
  res = tensordot(A, B, (indsA, indsB), final_order)
  print('BM 1- U1: {}s'.format(time.time() - t1))


def benchmark_1_U1xU1():

  R = 6
  charges = [
      U1Charge(
          np.asarray([-2, -1, -1, 0, -1, 0, 0, 1, -1, 0, 0, 1, 0, 1, 1, 2],
                     dtype=np.int16))
      @ U1Charge(
          np.asarray([0, -1, 1, 0, -1, -2, 0, -1, 1, 0, 2, 0, 0, -1, 1, 0],
                     dtype=np.int16)) for n in range(R)
  ]

  indsA = np.random.choice(np.arange(R), R // 2, replace=False)
  indsB = np.random.choice(np.arange(R), R // 2, replace=False)

  flowsA = np.asarray([False] * R)
  flowsB = np.asarray([False] * R)

  flowsB[indsB] = True
  A = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsA[n], name='a{}'.format(n)) for n in range(R)
  ])
  B = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsB[n], name='b{}'.format(n)) for n in range(R)
  ])

  final_order = np.arange(R)
  np.random.shuffle(final_order)
  t1 = time.time()
  res = tensordot(A, B, (indsA, indsB), final_order)
  print('BM 1- U1xU1: {}s'.format(time.time() - t1))


def benchmark_2_U1():
  R = 12
  charges = [
      U1Charge(np.asarray([-1, 0, 0, 1], dtype=np.int16)) for n in range(R)
  ]
  indsA = np.random.choice(np.arange(R), R // 2, replace=False)
  indsB = np.random.choice(np.arange(R), R // 2, replace=False)
  flowsA = np.asarray([False] * R)
  flowsB = np.asarray([False] * R)
  flowsB[indsB] = True
  A = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsA[n], name='a{}'.format(n)) for n in range(R)
  ])
  B = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsB[n], name='b{}'.format(n)) for n in range(R)
  ])
  final_order = np.arange(R)
  np.random.shuffle(final_order)
  t1 = time.time()
  res = tensordot(A, B, (indsA, indsB), final_order)
  print('BM 2- U1: {}s'.format(time.time() - t1))


def benchmark_2_U1xU1():
  R = 12
  charges = [
      U1Charge(np.asarray([-1, 0, 0, 1], dtype=np.int16)) @ U1Charge(
          np.asarray([0, -1, 1, 0], dtype=np.int16)) for n in range(R)
  ]
  indsA = np.random.choice(np.arange(R), R // 2, replace=False)
  indsB = np.random.choice(np.arange(R), R // 2, replace=False)
  flowsA = np.asarray([False] * R)
  flowsB = np.asarray([False] * R)
  flowsB[indsB] = True
  A = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsA[n], name='a{}'.format(n)) for n in range(R)
  ])
  B = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsB[n], name='b{}'.format(n)) for n in range(R)
  ])
  final_order = np.arange(R)
  np.random.shuffle(final_order)
  t1 = time.time()
  res = tensordot(A, B, (indsA, indsB), final_order)
  print('BM 2- U1xU1: {}s'.format(time.time() - t1))


def benchmark_3_U1():
  R = 14
  charges = [
      U1Charge(np.asarray([-1, 0, 0, 1], dtype=np.int16)) for n in range(R)
  ]

  indsA = np.random.choice(np.arange(R), R // 2, replace=False)
  indsB = np.random.choice(np.arange(R), R // 2, replace=False)
  flowsA = np.asarray([False] * R)
  flowsB = np.asarray([False] * R)
  flowsB[indsB] = True
  A = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsA[n], name='a{}'.format(n)) for n in range(R)
  ])
  B = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsB[n], name='b{}'.format(n)) for n in range(R)
  ])
  final_order = np.arange(R)
  np.random.shuffle(final_order)
  t1 = time.time()
  res = tensordot(A, B, (indsA, indsB), final_order)
  print('BM 3- U1: {}s'.format(time.time() - t1))


def benchmark_3_U1xU1():
  R = 14
  charges = [
      U1Charge(np.asarray([-1, 0, 0, 1], dtype=np.int16)) @ U1Charge(
          np.asarray([0, -1, 1, 0], dtype=np.int16)) for n in range(R)
  ]
  indsA = np.random.choice(np.arange(R), R // 2, replace=False)
  indsB = np.random.choice(np.arange(R), R // 2, replace=False)
  flowsA = np.asarray([False] * R)
  flowsB = np.asarray([False] * R)
  flowsB[indsB] = True
  A = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsA[n], name='a{}'.format(n)) for n in range(R)
  ])
  B = BlockSparseTensor.random(indices=[
      Index(charges[n], flowsB[n], name='b{}'.format(n)) for n in range(R)
  ])
  final_order = np.arange(R)
  np.random.shuffle(final_order)
  t1 = time.time()
  res = tensordot(A, B, (indsA, indsB), final_order)
  print('BM 3- U1xU1: {}s'.format(time.time() - t1))


benchmark_1_U1()
benchmark_1_U1xU1()
benchmark_2_U1()
benchmark_2_U1xU1()
benchmark_3_U1()
benchmark_3_U1xU1()
