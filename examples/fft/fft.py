"""Utility methods for producing interesting tensor networks."""

from typing import List
import numpy as np
import tensornetwork


def add_fft(
    net: tensornetwork.TensorNetwork,
    inputs: List[tensornetwork.Edge],
) -> List[tensornetwork.Edge]:
  """Creates output node axes corresponding to the Fourier transform of inputs.

  Uses Cooley-Tukey"s FFT algorithm. All axes are expected to have length 2. The
  input axes must be (and output axes will be) binary.

  Args:
    net: The tensor network to embed the fft network in.
    inputs: The node axes to act upon.

  Returns:
    A list of node axes containing the result.
  """
  if not all(e.is_dangling() for e in inputs):
    raise ValueError("Inputs must be dangling edges.")

  hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

  def cz(p: int) -> np.ndarray:
    result = np.eye(4, dtype=np.complex128)
    result[3, 3] = np.exp(-1j * np.pi / 2**p)
    return result.reshape((2,) * 4)

  def inline_stitch(targets: List[int], tensor: np.ndarray, name: str):
    """Applies an operation to the targeted axis indices."""
    op_node = net.add_node(tensor, name)
    for k, t in enumerate(targets):
      incoming_state = state[t]
      receiving_port = op_node[k]
      output_port = op_node[k + len(targets)]
      net.connect(incoming_state, receiving_port)
      state[t] = output_port

  state = list(inputs)

  # Mix "n twiddle.
  n = len(state)
  for i in range(n):
    for j in range(1, i + 1):
      inline_stitch([i - j, i], cz(j), "TWIDDLE_{}_{}".format(j, i))
    inline_stitch([i], hadamard, "MIX_{}".format(i))

  # FFT reverses bit order.
  return state[::-1]
