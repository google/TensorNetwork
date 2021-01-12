"""
Test cases for SU(2) Exact Diagonalization
"""
import numpy as np
from tensornetwork.ncon_interface import ncon
import Su2Engine
from Su2Tensor import FusionTree
import Su2ExactDiagonalization as SD
from scipy.sparse.linalg import eigs

########################################################################################################################
if __name__ == '__main__':

    ###### Test1 : diagonalize spin-1 Heisenberg model on a lattice of N=4 sites #######################################
    print('Test1: spin-1 Heisenberg model on a lattice of N=4 sites')
    N = 4 # number of lattice sites
    d = {1:1} # SU(2) charge on each lattice site = spin 1
    h = SD.Su2Create2SiteHamiltonianDensity('spin1Heisenberg')

    # full diagonalization. When running Su2ExactDiagonalization() for the first time for given N,d set doCreateTable
    # to True in order to generate the factor tables for use in subsequent calls (even for different ham, see Test2).
    # The table for N also work for any small M < N.
    spec = SD.Su2ExactDiagonalization(N,d,h,tableFileName = 'spinOneFactorTable.npy',doCreateTable = True)
    SD.printSpectrum(spec[0]) # prints sector-wise spectrum, but degeneracies are suppressed
    print('===========================================================')
    ###### End of Test1 : diagonalize spin-1 Heisenberg model on a lattice of N=4 sites ################################

    ###### Test2 : diagonalize spin-1 AKLT model on a lattice of N=4 sites #############################################
    print('Test2: spin-1 AKLT model on a lattice of N=4 sites')
    N = 4 # number of lattice sites
    d = {1:1} # SU(2) charge on each lattice site = spin 1
    h = SD.Su2Create2SiteHamiltonianDensity('aklt')

    # full diagonalization. Reuse factor table from Test1
    spec = SD.Su2ExactDiagonalization(N,d,h,tableFileName = 'spinOneFactorTable.npy',doCreateTable = False)
    SD.printSpectrum(spec[0])  # prints sector-wise spectrum, but degeneracies are suppressed

    # diagonalize to find 2 lowest states in total spin = 0, 3 states in total spin = 1, 2 states in total spin 2
    spec = SD.Su2ExactDiagonalization(N, d, h, targetSectors = {0:2, 1:3, 2:2}, tableFileName='spinOneFactorTable.npy',
                                                        doCreateTable=False) # reuse factor tables from previous runs
    SD.printSpectrum(spec[0])  # prints sector-wise spectrum, but degeneracies are suppressed
    print('===========================================================')
    ###### End of Test2 : diagonalize spin-1 AKLT model on a lattice of N=4 sites ######################################

    ###### Test3 : diagonalize spin-1/2 AFM Heisenberg model on a lattice of N=8 sites #################################
    print('Test2: spin-1/2 AFM Heisenberg model on a lattice of N=8 sites')
    N = 8  # number of lattice sites
    d = {0.5: 1}  # SU(2) charge on each lattice site = spin 1/2
    h = SD.Su2Create2SiteHamiltonianDensity('afmHeisenberg')

    # full diagonalization. Create factor tables for spin 1/2 chain with N=8 sites and store in specified file.
    spec = SD.Su2ExactDiagonalization(N, d, h, tableFileName='spinHalfFactorTable.npy', doCreateTable=True)
    SD.printSpectrum(spec[0])  # prints sector-wise spectrum, but degeneracies are suppressed
    print('===========================================================')
    ###### End of Test3 : diagonalize spin-1 AKLT model on a lattice of N=4 sites ######################################

    ###### Test4 : Calculate Wigner and Clebsch-Gordan coefficients of SU(2) ###########################################
    print('Demonstration of functions to calculate Wigner and Clebsch-Gordan coefficients of SU(2)')
    w3j = Su2Engine.wigner3j(0.5, 0.5, 0.5, -0.5, 0, 0)
    cg = Su2Engine.clebschgordan(0.5, 0.5, 0.5, -0.5, 0, 0)
    cg1 = Su2Engine.clebschgordan(1, -1, 1, 0, 1, -1)
    cg2 = Su2Engine.clebschgordan(1, -1, 1, 0, 2, -2)
    C = Su2Engine.createCGtensor(0.5, 0.5, 1)
    factor = Su2Engine.fsymbol(1, 1, 1, 1, 1, 1)
    factor1 = Su2Engine.fsymbol(1, 2, 0.5, 3, 1.5, 2.5)
    print('===========================================================')
    ###### End of Test4 : Calculate Wigner and Clebsch-Gordan coefficients of SU(2) ####################################

    ###### Test5 : Check fusion rules of SU(2) #########################################################################
    print('Demonstration of functions to check fusion rules of SU(2)')
    res = Su2Engine.compatible(j1=0.5, j2=0.5, j12=0.5)
    res = Su2Engine.compatible(j1=1, j2=1, j12=1)
    res = Su2Engine.compatible(j1=1, j2=0.5, j12=1)
    print('===========================================================')
    ###### End of Test5 : Check fusion rules of SU(2) ##################################################################

    ###### Test5 : Fusing two SU(2) indices ############################################################################
    print('Demonstration of fusing two SU(2) indices')
    ia = {0.5:1}
    ib = {0.5:1}
    ic = {0.5:3, 2.5:1.3}

    try:
        Su2Engine.isvalidindex(ia)
    except:
        print('Invalid index')

    try:
        Su2Engine.isvalidindex(ic)
    except:
        print('Invalid index')

    iab = Su2Engine.fuseindices(ia,ib) # fuses indices ia and ib. index iab has the fused spin and degeneracy values
    iiab = Su2Engine.fuseindices(iab,iab) # another example

    ia = {0:1, 1:1}
    ib = {0:1, 1:1}
    X = Su2Engine.fuse(ia,ib,[0,1]) # X is the fusion tensor. Its a Su2Tensor that can be contracted with another
                                    # Su2Tensor inorder to fuse two indices and reshape that tensor
    block000 = X.getblock(FusionTree((0, 0, 0))) # should be some assignments of 0's and 1's to give an injective map
                                                 # from input indices to fused index
    block011 = X.getblock(FusionTree((0, 1, 1)))
    block101 = X.getblock(FusionTree((1, 0, 1)))
    block110 = X.getblock(FusionTree((1, 1, 0)))
    block111 = X.getblock(FusionTree((1, 1, 1)))

    ia = {0.5:1, 1:2, 3:4}
    ib = {0.5:1, 5:5, 2:1}
    X = Su2Engine.fuse(ia,ib)
    block0 = X.getblock(FusionTree((0, 0.5, 0.5)))
    block1 = X.getblock(FusionTree((1, 1, 2)))
    print('===========================================================')
    ###### End of Test5 : Fusing two SU(2) indices #####################################################################
