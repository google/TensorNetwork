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

""" Performs exact diagonalization of SU(2)-symmetric Hamiltonians by exploiting the symmetry.
    - Diagonalization can be restricted to specified total spin sectors.
    - Uses pre-computation of contraction factors resulting from contracting Clebsch-Gordan tensors;
      factors are computed on first run of the code and stored in a file.
    - Provides a preset of popular SU(2)-symmetric Hamiltonian densities
    - Assumptions: 1) Hamiltonian density must be a 2-site operator. 2) Presently only implements open
                   boundary condition. 3) Only uses ncon() facility of the TensorNetwork library."""

import numpy as np
from numpy import linalg as LA
from math import floor
from scipy.sparse.linalg import eigs

from tensornetwork.ncon_interface import ncon

from Su2Tensor import Su2Tensor
from Su2Tensor import FusionTree
import Su2Engine

# globals
total, left, right = (0, 1, 2)  # some constants (index values)

########################################################################################################################
def Su2ExactDiagonalization(
        N: 'int', d: 'dict', h: Su2Tensor, targetSectors: list=[],
        tableFileName: 'string' = 'FT_Su2CoarseGrain2To1Site.npy',
        doCreateTable: 'bool' =False) -> tuple[dict, dict, np.array]:
    """
    Performs exact diagonalization in specified targetSectors. Returns the low-energy states and their energies in
    these sectors.
    N: number of lattice sites.
    d: SU(2) index describing each lattice site. Must be a dictionary of the type {total spin: degeneracy dimension}
    h: 2-site Hamiltonian density. Must be an object of SU2Tensor class
    targetSectors: Specifies which total spin sectors (of the entire lattice) to diagonalize and how many states
                   required in each sector. Must be dictionary of the type: {total spin, number of states}
                   If None diagonalizes all the sectors
    doCreateTable: set to True if the factor tables required by this function have not been created yet.
    tableFileName: file to store the table when table creation is demanded.
    """

    # check validity of inputs
    if not Su2Engine.myisinteger(N):
        raise ValueError('Lattice size must be a natural number!')
    if N == 0 or N == 1:
        raise ValueError('Lattice size must be greater than equal to 2!')
    Su2Engine.isvalidindex(d)
    if not isinstance(h,Su2Tensor):
        raise ValueError('Hamiltonian density must be a SU(2) tensor.')

    # create blocking isometries from bottom (physical index) to top(total index)
    w = [] # list of blocking isometries
    bond = d
    for k in range(N-2):
        X = Su2Engine.fuse(bond, d)
        w.append(X)
        bond = X.getIndices()[0] # set bond to new total index

    # Create the final blocking isometry specially
    if targetSectors: # the user has asked to truncate final index
        X = Su2Engine.fuse(bond, d, list(targetSectors.keys())) # truncate total spin sectors
    else:
        X = Su2Engine.fuse(bond, d) # otherwise the full blocking tensor
    w.append(X)
    bond = X.getIndices()[0] # bond is now the total spin basis on the entire lattice

    if doCreateTable:
        factorTable = createFactorTable_Su2CoarseGrain_2To1Site([[w[N-4],w[N-3]], [w[N-3],w[N-2]]],w[0], tableFileName)
    else:
        factorTable = np.load(tableFileName, allow_pickle='TRUE').item()

    su2H = Su2Tensor([bond, bond]) # Total Hamiltonian.
    wd = w[0]
    for term in range(N-1):
        # first coarse-grain local term to a 1-site operator. Fist term has to
        # be treated differently from the others.

        if term == 0:
            hh = h # already blocked to a one-site operator
        else:
            hh = Su2CoarseGrain_2To1site(w[term-1], w[term], wd, h, factorTable)

        # now coarse-grain this one-site operator through the remaining linear fusion tree
        for m in range(term+1,N-1):
            hh = Su2CoarseGrain_1To1site(w[m],hh)

        # hh is now the one-site version of the local hamiltonian term. Add to current sum
        hhBlocks = hh.getAllBlocks()
        for fusionTree in hhBlocks:
            su2H.addblock(fusionTree,hhBlocks[fusionTree])

    # Now let's diagonalize
    spectrum = dict() # dictionary of the type: {total spin: spectra}
    eigenvectors = dict() # dictionary of the type: {total spin: eigenvectors}
    fullspectrum = None # full spectrum (including degneracies) as a NumPy array
    su2HBlocks = su2H.getAllBlocks()
    for fusionTree in su2HBlocks:
        degdim = 2*fusionTree[0]+1
        if targetSectors: # diagonalize only the sectors that the user wants
            temp, ev = eigs(su2HBlocks[fusionTree], k=targetSectors[fusionTree[0]], which='SR')
        else: # do full digonalization of all the blocks
            temp = LA.eigvals(su2HBlocks[fusionTree]) # full diagonalization
            ev = None
        spectrum[(fusionTree[0])] = temp
        eigenvectors[(fusionTree[0])] = ev
        temp = np.kron(temp,np.ones((1,floor(degdim))))
        if fullspectrum is None:
            fullspectrum = temp
        else:
            fullspectrum = np.append(fullspectrum, temp)
    fullspectrum.sort()

    return spectrum, eigenvectors, fullspectrum

########################################################################################################################
def Su2CoarseGrain_2To1site(
        w1: Su2Tensor, w2: Su2Tensor, wd: Su2Tensor, h: Su2Tensor, factorTable: dict) -> Su2Tensor:
    """
    Coarse-grains 2-site Ham h to a 1-site term through isometries w1,w2 inside a linear fusion tree
    w2 is the blocking isometry from the higher level. h,hL,w1,w2 are SU(2)-symmetric tensors.
    """

    hh = Su2Tensor(h.getIndices())  # the coarse-grained hamiltonian. Even though h is a 4-index tensor it is
                                    # stored as a 2-index matrix. So indices of h are the blocked 2-site indices
    w2Blocks = w2.getAllBlocks()
    w1Blocks = w1.getAllBlocks()
    wdBlocks = wd.getAllBlocks()
    hBlocks = h.getAllBlocks()
    for ft in factorTable: # iterate through all fusion trees in table
        c1 = FusionTree((ft[0], ft[1], ft[4]))
        c2 = FusionTree((ft[1], ft[2], ft[3]))
        c3 = FusionTree((ft[5], ft[3], ft[4]))
        c4 = FusionTree((ft[5], ft[6], ft[7]))
        c5 = FusionTree((ft[8], ft[2], ft[6]))
        c6 = FusionTree((ft[0], ft[8], ft[7]))
        ch = FusionTree((ft[5], ft[5]))

        if not c1 in w2Blocks or not c2 in w1Blocks or not c3 in wdBlocks or not c4 in wdBlocks or not c5 in w1Blocks \
                or not c6 in w2Blocks or not ch in hBlocks:
            continue
        temp = ncon([w2Blocks[c1],w1Blocks[c2],wdBlocks[c3],hBlocks[ch],wdBlocks[c4],w1Blocks[c5],w2Blocks[c6]],\
                    [[-1,2,4],[2,8,3],[1,3,4],[1,9],[9,6,7],[5,8,6],[-2,5,7]])
        hh.addblock(FusionTree((ft[0],ft[0])),factorTable[ft]*temp)

    return hh

########################################################################################################################
def Su2CoarseGrain_1To1site(
    w: Su2Tensor, h: Su2Tensor) -> Su2Tensor:
    """
    Coarse-grains (blocks) 1-site operator h through isometry w.
    w: the blocking isometry
    h: the 1-site SU(2)-symmetric operator
    """

    hh = Su2Tensor(h.getIndices()) # the coarse-grained hamiltonian. Even though h is a 4-index tensor it is
                                   # stored as a 2-index matrix. So indices of h are the blocked 2-site indices
    wBlocks = w.getAllBlocks() # a dictionary {FusionTree:block}
    hBlocks = h.getAllBlocks() # a dictionary {FusionTree:block}
    for ft in wBlocks:
        ch = FusionTree((ft[1],ft[1]))
        if not ch in hBlocks:
            continue
        temp = ncon([wBlocks[ft],hBlocks[ch],wBlocks[ft]],[[-1, 1, 3],[1, 2],[-2, 2, 3]])
        hh.addblock(FusionTree((ft[0],ft[0])),temp)

    return hh

########################################################################################################################
def createFactorTable_Su2CoarseGrain_2To1Site(
        W: 'list of pairs of Su2Tensors', wd: Su2Tensor, tableFileName: str) -> dict:
    """
    (Precomputation.) Creates the factors and compatible chargesectors for the function Su2CoarseGrain_2To1Site()
    W: a list of pairs of SU(2) tensors
    wd: a SU(2) tensor (the isometry that blocks 2 lattice sites)
    tableFileName: file name to store the created factor table
    """

    FactorTable = dict() # Dictionary from fusion trees to factors
    for k in range(len(W)):
        w1 = W[k][0]
        w2 = W[k][1]
        for c1 in w2.getAllBlocks():
            for c2 in w1.getAllBlocks():
                if c2[total] != c1[left]:
                    continue
                for c3 in wd.getAllBlocks():
                    if c2[right] != c3[left] or c1[right] != c3[right] or not Su2Engine.compatible(c2[left], c3[total],\
                                                                                                             c1[total]):
                        continue
                    for c4 in wd.getAllBlocks():
                        if c4[total] != c3[total]:
                            continue
                        for c5 in w1.getAllBlocks():
                            if c5[right] != c4[left] or c5[left] != c2[left] or not Su2Engine.compatible(c5[total],\
                                                                                                c4[right],c1[total]):
                                continue
                            for c6 in w2.getAllBlocks():
                                if c6[left] != c5[total] or c6[right] != c4[right] or c6[total] != c1[total]:
                                    continue

                                C1 = Su2Engine.createCGtensor(c1[left], c1[right], c1[total])
                                C2 = Su2Engine.createCGtensor(c2[left], c2[right], c2[total])
                                C3 = Su2Engine.createCGtensor(c3[left], c3[right], c3[total])
                                C4 = Su2Engine.createCGtensor(c4[left], c4[right], c4[total])
                                C5 = Su2Engine.createCGtensor(c5[left], c5[right], c5[total])
                                C6 = Su2Engine.createCGtensor(c6[left], c6[right], c6[total])
                                Cmat = ncon([C1,C2,C3,C4,C5,C6],[[1,3,-1],[7,2,1],[2,3,8],[5,6,8],[7,5,4],[4,6,-2]])
                                FactorTable[FusionTree((c1[total],c1[left],c2[left],c2[right],c1[right],c3[total],\
                                                                           c4[left],c4[right],c5[total]))] = Cmat[0][0]

    np.save(tableFileName, FactorTable)
    return FactorTable

########################################################################################################################
def Su2Create2SiteHamiltonianDensity(
        whichHam: str) -> Su2Tensor:
    """
    Creates 2-site Hamiltonian density as a SU(2) tensor. d is the SU(2) index for each lattice site.
    whichHam is a string to select from a preset of SU(2)-symmetry Hamiltonian densities:
    'afmHeisenberg', 'spin1Heisenberg', 'aklt'"""
    if whichHam == 'afmHeisenberg':
        d2 = {0:1,1:1}
        h = Su2Tensor([d2, d2])
        block0 = np.ones((1,1))
        block0[0][0] = -3
        block1 = np.ones((1,1))
        h.addblock(FusionTree((0, 0)), block0)
        h.addblock(FusionTree((1, 1)), block1)
    elif whichHam == 'identity': # for testing purposes
        d2 = {0: 1, 1: 1}
        h = Su2Tensor([d2, d2])
        block0 = np.ones((1, 1))
        block1 = np.ones((1, 1))
        h.addblock(FusionTree((0, 0)), block0)
        h.addblock(FusionTree((1, 1)), block1)
    elif whichHam == 'spin1Heisenberg':
        d2 = {0: 1, 1: 1, 2: 1}
        h = Su2Tensor([d2, d2])
        block0 = np.ones((1, 1))
        block0[0][0] = -2
        block1 = np.ones((1, 1))
        block1[0][0] = -1
        block2 = np.ones((1, 1))
        block2[0][0] = 1
        h.addblock(FusionTree((0, 0)), block0)
        h.addblock(FusionTree((1, 1)), block1)
        h.addblock(FusionTree((2, 2)), block2)
    elif whichHam == 'aklt':
        d2 = {0: 1, 1: 1, 2: 1}
        h = Su2Tensor([d2, d2])
        block0 = np.ones((1, 1))
        block0[0][0] = -2/3
        block1 = np.ones((1, 1))
        block1[0][0] = -2/3
        block2 = np.ones((1, 1))
        block2[0][0] = 4/3
        h.addblock(FusionTree((0, 0)), block0)
        h.addblock(FusionTree((1, 1)), block1)
        h.addblock(FusionTree((2, 2)), block2)
    elif whichHam == 'blbqHeisenberg':
        pass

    return h
