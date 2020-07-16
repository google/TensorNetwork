import time

def tick():
  return time.perf_counter()


def tock(t0, dat=None):
  return time.perf_counter() - t0
