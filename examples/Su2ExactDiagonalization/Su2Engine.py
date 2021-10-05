# Copyright 2020 The TensorNetwork Authors
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

"""Basic SU(2) representation theory data"""

from math import floor
from math import factorial
from math import sqrt
from collections import defaultdict
import numpy as np
from tensornetwork.ncon_interface import ncon
from Su2Tensor import Su2Tensor
from Su2Tensor import FusionTree

def myisinteger(
        num: int) -> bool:
    """
    Checks if num is an integer
    """

    val = 1 if num == floor(num) else 0
    return val

#######################################################################################
def compatible(
        j1: float, j2: float, j12: float) -> bool:
    """
    Checks if total spins j1,j2,j12 are compatible with SU(2) fusion rules
    """

    val = True if j12 in np.arange(abs(j1-j2),j1+j2+1,1).tolist() else False
    return val

#######################################################################################
def wigner3j(
        j1: float, m1: float, j2: float, m2: float, j12: float, m12: float) -> float:
    """
    Computes the Wigner 3j symbol. j's at the total spin values and m's are the
    corresponding spin projection values
    """
    # compute Wigner 3j symbol
    if not myisinteger(2*j1) or not myisinteger(2*j2) or not myisinteger(2*j12) \
            or not myisinteger(2*m1) or not myisinteger(2*m2) or not myisinteger(2*m12):
        raise ValueError('Input values must be (semi-)integers!')
    if not myisinteger(2*(j1-m1)):
        raise ValueError('j1-m1 must be an integer!')
    if not myisinteger(2*(j2-m2)):
        raise ValueError('j2-m2 must be an integer!')
    if not myisinteger(2*(j12-m12)):
        raise ValueError('j12-m12 must be an integer!')
    if not compatible(j1,j2,j12):
        raise ValueError('The spins j1,j2,j12 are not compatible with the fusion rules!')
    if abs(m1) > j1:
        raise ValueError('m1 is out of bounds.')
    if abs(m2) > j2:
        raise ValueError('m2 is out of bounds.')
    if abs(m12) > j12:
        raise ValueError('m12 is out of bounds.')
    if m1 + m2 + m12 != 0:
        return 0    # m's are not compatible. So coefficient must be 0.
    t1 = j2 - m1 - j12
    t2 = j1 + m2 - j12
    t3 = j1 + j2 - j12
    t4 = j1 - m1
    t5 = j2 + m2

    tmin = max([0, t1, t2])
    tmax = min([t3, t4, t5])

    wigner = 0
    for t in np.arange(tmin, tmax+1, 1).tolist():
        wigner = wigner + (-1)**t / (factorial(t) * factorial(t - t1) * factorial(t - t2)\
                  * factorial(t3 - t) * factorial(t4 - t) * factorial(t5 - t))

    wigner = wigner * (-1)**(j1-j2-m12) * sqrt(factorial(j1+j2-j12) * factorial(j1-j2+j12)\
              * factorial(-j1+j2+j12) / factorial(j1+j2+j12+1) * factorial(j1+m1)\
              * factorial(j1-m1) * factorial(j2+m2) * factorial(j2-m2) * factorial(j12+m12)\
              * factorial(j12-m12))

    return wigner

#######################################################################################
def clebschgordan(
        j1: float, m1: float, j2: float, m2: float, j12: float, m12: float) -> float:
    """
    Computes a Clebsch-Gordan coefficient. j's at the total spin values and m's are the corresponding
    spin projection values
    """
    return (-1)**(j1-j2+m12) * sqrt(2*j12+1) * wigner3j(j1,m1,j2,m2,j12,-m12) # Clebsch-Gordan coefficient from Wigner 3j

#######################################################################################
def createCGtensor(
        j1: float, j2: float, j12: float) -> np.array:
    """
    Computes a Clebsch-Gordan tensor for the fusion j1xj2 -> j12
    """

    C = np.zeros((floor(2*j1+1), floor(2*j2+1), floor(2*j12+1)))
    if not compatible(j1,j2,j12):
        return C
    for m1 in np.arange(-j1, j1+1, 1).tolist():
        for m2 in np.arange(-j2, j2+1, 1).tolist():
            for m12 in np.arange(-j12, j12+1, 1).tolist():
                C[floor(j1+m1)][floor(j2+m2)][floor(j12+m12)] = clebschgordan(j1,m1,j2,m2,j12,m12) if m1+m2 == m12 else 0
    return C

#######################################################################################
def rsymbol(
        j1: float, j2: float, j12: float) -> float:
    """
    Returns a R-symbol of SU(2)
    """

    C1 = createCGtensor(j1,j2,j12)
    C2 = createCGtensor(j2,j1,j12)
    C = ncon([C1, C2],[[1, 2, -1],[2, 1, -2]])
    return C[0][0]

#######################################################################################
def racahDenominator(
        t: float, j1: float, j2: float, j3: float, J1: float, J2: float, J3: float) -> float:
    """
    Helper function used in wigner6j().
    """

    return factorial(t-j1-j2-j3) * factorial(t-j1-J2-J3) * factorial(t-J1-j2-J3) * factorial(t-J1-J2-j3) \
    * factorial(j1+j2+J1+J2-t) * factorial(j2+j3+J2+J3-t) * factorial(j3+j1+J3+J1-t)

#######################################################################################
def triangleFactor(
        a: int, b: int, c: int) -> float:
    """
    Helper function used in wigner6j().
    """

    return factorial(a+b-c) * factorial(a-b+c) * factorial(-a+b+c)/factorial(a+b+c+1)

#######################################################################################
def wigner6j(
        j1: float, j2: float, j3: float, J1: float, J2: float, J3: float) -> float:
    """
    Returns a Wigner 6j-symbol of SU(2).
    """

    tri1 = triangleFactor(j1, j2, j3)
    tri2 = triangleFactor(j1, J2, J3)
    tri3 = triangleFactor(J1, j2, J3)
    tri4 = triangleFactor(J1, J2, j3)

    # Calculate the summation range in Racah formula
    tmin = max([j1+j2+j3, j1+J2+J3, J1+j2+J3, J1+J2+j3])
    tmax = min([j1+j2+J1+J2, j2+j3+J2+J3, j3+j1+J3+J1])

    Wigner = 0

    for t in np.arange(tmin, tmax+1, 1).tolist():
        Wigner = Wigner + sqrt(tri1*tri2*tri3*tri4) * ((-1)**t) * factorial(t+1)/racahDenominator(t,j1,j2,j3,J1,J2,J3)

    return Wigner

#######################################################################################
def fsymbol(
        j1: float, j2: float, j3: float, j12: float, j23: float, j: float) -> float:
    """
    Returns he recouping coefficient of SU(2), related to the Wigner 6j symbol by an overall factor.
    """

    if not compatible(j1,j2,j12) or not compatible(j2,j3,j23) or not compatible(j1,j23,j) or not compatible(j12,j3,j):
        return 0

    return wigner6j(j1,j2,j12,j3,j,j23) * sqrt(2*j12+1) * sqrt(2*j23+1) * (-1)**(j1+j2+j3+j)

#######################################################################################
def isvalidindex(
        index: dict) -> None:
    """
    Checks for a valid SU(2) index, and raises an exception if its not.
    index: a Dictionary of the type {total spin value: degeneracy dimension}.
    """

    if not isinstance(index, dict):
        raise ValueError('SU(2) index should be a dictionary of the type: {total spin, degeneracy dimension}')

    for key in index:
        if not myisinteger(2*key):
            raise ValueError('Spin values in an index should be (semi-)integers.')
        if not myisinteger(index[key]):
            raise ValueError('Degeneracy values in an index should be integers.')

#########################################################################################
def fuseindices(
        ia: dict, ib: dict) -> dict:
    """
    Fuses two SU(2) indices ia, ib according to the fusion rules.
    ia, ib: Dictionaries of the type {total spin value: degeneracy dimension}.
    """

    if len(ia) == 0 or len(ib) == 0:
        raise ValueError('Input indices cannot be empty.')
    isvalidindex(ia)
    isvalidindex(ib)

    iab = defaultdict(int) # initialize fused index to empty default dictionary
    for ja in ia:
        da = ia[ja]
        for jb in ib:
            db = ib[jb]
            for jjab in np.arange(abs(ja-jb), ja+jb+1, 1).tolist():
                if len(iab) == 0:
                    iab[jjab] = da*db
                else:
                    iab[jjab] += da*db

    return iab

########################################################################################
def fuse(
        ia: dict, ib: dict, totalSectors: list=[]) -> Su2Tensor:
    """
    Creates a 3-index tensor X that two SU(2) indices according to the fusion rules.
    ia, ib: Dictionaries of the type {total spin value: degeneracy dimension}.
    totalSectors (optional): list of total sectors to keep. [total spin values]
    """
    iab = fuseindices(ia, ib)

    # remove some total sectors if targetSectors are specified
    if totalSectors:
        iabCopy = dict(iab) # shallow copy should suffice
        for jab in iab:
            if not jab in totalSectors:
                iabCopy.pop(jab) # remove unwanted total sectors
        iab = iabCopy # update iab

    dabGone = defaultdict(int) # dictionary index by jab to keep track to degeneracies as they are filled in X

    X = Su2Tensor([iab, ia, ib])
    for ja in ia:
        da = ia[ja]
        for jb in ib:
            db = ib[jb]
            for jab in iab:
                if not compatible(ja,jb,jab):
                    continue
                dab = iab[jab]
                temp = np.zeros((da*db,dab))
                temp[0:db*da, dabGone[jab]:db*da+dabGone[jab]] = np.identity(db*da)
                temp = temp.reshape((db, da, dab), order='F')  # Specifying order=F means a column-major reshape, similar to MATLAB. Default in python is row-major
                temp = temp.transpose(2,1,0)
                X.addblock(FusionTree((jab, ja, jb)),temp)
                dabGone[jab] += da*db

    return X