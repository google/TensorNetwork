# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#pylint: disable=line-too-long
from typing import Optional, Sequence, Tuple, Any, Union, Type, Callable, List, Text
import numpy as np
import tensornetwork.tensor
import tensornetwork.backends.abstract_backend as abstract_backend
from tensornetwork import backend_contextmanager
from tensornetwork import backends

AbstractBackend = abstract_backend.AbstractBackend
Array = Any
Tensor = tensornetwork.tensor.Tensor

def eigsh_lanczos(A: Callable,
                  backend: Optional[Union[Text, AbstractBackend]] = None,
                  args: Optional[List[Tensor]] = None,
                  initial_state: Optional[Tensor] = None,
                  shape: Optional[Tuple[int, ...]] = None,
                  dtype: Optional[Type[np.number]] = None,
                  num_krylov_vecs: int = 20,
                  numeig: int = 1,
                  tol: float = 1E-8,
                  delta: float = 1E-8,
                  ndiag: int = 20,
                  reorthogonalize: bool = False) -> Tuple[Tensor, List]:
  """
  Lanczos method for finding the lowest eigenvector-eigenvalue pairs
  of `A`.
  Args:
    A: A (sparse) implementation of a linear operator.
       Call signature of `A` is `res = A(vector, *args)`, where `vector`
       can be an arbitrary `Array`, and `res.shape` has to be `vector.shape`.
    arsg: A list of arguments to `A`.  `A` will be called as
      `res = A(initial_state, *args)`.
    initial_state: An initial vector for the Lanczos algorithm. If `None`,
      a random initial vector is created using the `backend.randn` method
    shape: The shape of the input-dimension of `A`.
    dtype: The dtype of the input `A`. If both no `initial_state` is provided,
      a random initial state with shape `shape` and dtype `dtype` is created.
    num_krylov_vecs: The number of iterations (number of krylov vectors).
    numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
      If `numeig > 1`, `reorthogonalize` has to be `True`.
    tol: The desired precision of the eigenvalus. Uses
      `backend.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
      as stopping criterion between two diagonalization steps of the
      tridiagonal operator.
    delta: Stopping criterion for Lanczos iteration.
      If a Krylov vector :math: `x_n` has an L2 norm
      :math:`\\lVert x_n\\rVert < delta`, the iteration
      is stopped. It means that an (approximate) invariant subspace has
      been found.
    ndiag: The tridiagonal Operator is diagonalized every `ndiag`
      iterations to check convergence.
    reorthogonalize: If `True`, Krylov vectors are kept orthogonal by
      explicit orthogonalization (more costly than `reorthogonalize=False`)
  Returns:
    (eigvals, eigvecs)
     eigvals: A list of `numeig` lowest eigenvalues
     eigvecs: A list of `numeig` lowest eigenvectors
  """
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend = backends.backend_factory.get_backend(backend)
  if args is not None:
    args = [a.array for a in args]
  if initial_state is not None:
    initial_state = initial_state.array
  result = backend.eigsh_lanczos(A, args=args,
                                 initial_state=initial_state,
                                 shape=shape, dtype=dtype,
                                 num_krylov_vecs=num_krylov_vecs, numeig=numeig,
                                 tol=tol, delta=delta, ndiag=ndiag,
                                 reorthogonalize=reorthogonalize)
  eigvals, eigvecs = result
  eigvecsT = [Tensor(ev, backend=backend) for ev in eigvecs]
  return eigvals, eigvecsT


def eigs(A: Callable,
         backend: Optional[Union[Text, AbstractBackend]] = None,
         args: Optional[List[Tensor]] = None,
         initial_state: Optional[Tensor] = None,
         shape: Optional[Tuple[int, ...]] = None,
         dtype: Optional[Type[np.number]] = None,
         num_krylov_vecs: int = 20,
         numeig: int = 1,
         tol: float = 1E-8,
         which: Text = 'LR',
         maxiter: int = 20) -> Tuple[Tensor, List]:
  """
  Lanczos method for finding the lowest eigenvector-eigenvalue pairs
  of `A`.
  Args:
    A: A (sparse) implementation of a linear operator.
       Call signature of `A` is `res = A(vector, *args)`, where `vector`
       can be an arbitrary `Array`, and `res.shape` has to be `vector.shape`.
    arsg: A list of arguments to `A`.  `A` will be called as
      `res = A(initial_state, *args)`.
    initial_state: An initial vector for the Lanczos algorithm. If `None`,
      a random initial vector is created using the `backend.randn` method
    shape: The shape of the input-dimension of `A`.
    dtype: The dtype of the input `A`. If both no `initial_state` is provided,
      a random initial state with shape `shape` and dtype `dtype` is created.
    num_krylov_vecs: The number of iterations (number of krylov vectors).
    numeig: The nummber of eigenvector-eigenvalue pairs to be computed.
      If `numeig > 1`, `reorthogonalize` has to be `True`.
    tol: The desired precision of the eigenvalus. Uses
      `backend.norm(eigvalsnew[0:numeig] - eigvalsold[0:numeig]) < tol`
      as stopping criterion between two diagonalization steps of the
      tridiagonal operator.
    delta: Stopping criterion for Lanczos iteration.
      If a Krylov vector :math: `x_n` has an L2 norm
      :math:`\\lVert x_n\\rVert < delta`, the iteration
      is stopped. It means that an (approximate) invariant subspace has
      been found.
    ndiag: The tridiagonal Operator is diagonalized every `ndiag`
      iterations to check convergence.
    reorthogonalize: If `True`, Krylov vectors are kept orthogonal by
      explicit orthogonalization (more costly than `reorthogonalize=False`)
  Returns:
    (eigvals, eigvecs)
     eigvals: A list of `numeig` lowest eigenvalues
     eigvecs: A list of `numeig` lowest eigenvectors
  """
  if backend is None:
    backend = backend_contextmanager.get_default_backend()
  backend = backends.backend_factory.get_backend(backend)
  if args is not None:
    args = [a.array for a in args]
  if initial_state is not None:
    initial_state = initial_state.array
  result = backend.eigs(A, args=args, initial_state=initial_state,
                        shape=shape, dtype=dtype,
                        num_krylov_vecs=num_krylov_vecs, numeig=numeig,
                        tol=tol, which=which, maxiter=maxiter)
  eigvals, eigvecs = result
  eigvecsT = [Tensor(eV, backend=backend) for eV in eigvecs]
  return eigvals, eigvecsT


def gmres(A_mv: Callable,
          b: Tensor,
          backend: Optional[Union[Text, AbstractBackend]] = None,
          A_args: Optional[List] = None,
          x0: Optional[Tensor] = None,
          tol: float = 1E-05,
          atol: Optional[float] = None,
          num_krylov_vectors: Optional[int] = None,
          maxiter: Optional[int] = 1,
          M: Optional[Callable] = None
          ) -> Tuple[Tensor, int]:
  """ GMRES solves the linear system A @ x = b for x given a vector `b` and
  a general (not necessarily symmetric/Hermitian) linear operator `A`.

  As a Krylov method, GMRES does not require a concrete matrix representation
  of the n by n `A`, but only a function
  `vector1 = A_mv(vector0, *A_args, **A_kwargs)`
  prescribing a one-to-one linear map from vector0 to vector1 (that is,
  A must be square, and thus vector0 and vector1 the same size). If `A` is a
  dense matrix, or if it is a symmetric/Hermitian operator, a different
  linear solver will usually be preferable.

  GMRES works by first constructing the Krylov basis
  K = (x0, A_mv@x0, A_mv@A_mv@x0, ..., (A_mv^num_krylov_vectors)@x_0) and then
  solving a certain dense linear system K @ q0 = q1 from whose solution x can
  be approximated. For `num_krylov_vectors = n` the solution is provably exact
  in infinite precision, but the expense is cubic in `num_krylov_vectors` so
  one is typically interested in the `num_krylov_vectors << n` case.
  The solution can in this case be repeatedly
  improved, to a point, by restarting the Arnoldi iterations each time
  `num_krylov_vectors` is reached. Unfortunately the optimal parameter choices
  balancing expense and accuracy are difficult to predict in advance, so
  applying this function requires a degree of experimentation.

  In a tensor network code one is typically interested in A_mv implementing
  some tensor contraction. This implementation thus allows `b` and `x0` to be
  of whatever arbitrary, though identical, shape `b = A_mv(x0, ...)` expects.
  Reshaping to and from a matrix problem is handled internally.

  Args:
    A_mv     : A function `v0 = A_mv(v, *A_args, **A_kwargs)` where `v0` and
               `v` have the same shape.
    b        : The `b` in `A @ x = b`; it should be of the shape `A_mv`
               operates on.
    backend  : The backend.
    A_args   : Positional arguments to `A_mv`, supplied to this interface
               as a list.
               Default: None.
    x0       : An optional guess solution. Zeros are used by default.
               If `x0` is supplied, its shape and dtype must match those of
               `b`, or an
               error will be thrown.
               Default: zeros.
    tol, atol: Solution tolerance to achieve,
               norm(residual) <= max(tol*norm(b), atol).
               Default: tol=1E-05
                        atol=tol
    num_krylov_vectors
             : Size of the Krylov space to build at each restart.
               Expense is cubic in this parameter. If supplied, it must be
               an integer in 0 < num_krylov_vectors <= b.size.
               Default: b.size.
    maxiter  : The Krylov space will be repeatedly rebuilt up to this many
               times. Large values of this argument
               should be used only with caution, since especially for nearly
               symmetric matrices and small `num_krylov_vectors` convergence
               might well freeze at a value significantly larger than `tol`.
               Default: 1.
    M        : Inverse of the preconditioner of A; see the docstring for
               `scipy.sparse.linalg.gmres`. This is only supported in the
               numpy backend. Supplying this argument to other backends will
               trigger NotImplementedError.
               Default: None.

  Raises:
    ValueError: -if `x0` is supplied but its shape differs from that of `b`.
                -in NumPy, if the ARPACK solver reports a breakdown (which
                 usually indicates some kind of floating point issue).
                -if num_krylov_vectors is 0 or exceeds b.size.
                -if tol was negative.
                -if M was supplied with any backend but NumPy.

  Returns:
    x       : The converged solution. It has the same shape as `b`.
    info    : 0 if convergence was achieved, the number of restarts otherwise.
  """
  if A_args is not None:
    A_args = [a.array for a in A_args]
  if x0 is not None:
    x0 = x0.array
  out = b.backend.gmres(A_mv, b.array, A_args=A_args, 
                        x0=x0, tol=tol, atol=atol,
                        num_krylov_vectors=num_krylov_vectors,
                        maxiter=maxiter, M=M)
  result, info = out
  resultT = Tensor(result, backend=b.backend)
  return (resultT, info)
