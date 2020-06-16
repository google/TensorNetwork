

import numpy as np
from solve_functs import ncon_solver
from tensornetwork import ncon

# define a network
chi = 4;
chi_p = 3;
A = np.random.rand(chi,chi,chi,chi)
B = np.random.rand(chi,chi_p,chi_p)
tensors = [A,A,A,A,A,A,A,A,A,A,B]
connects = [[-1,-2,1,2],[1,2,3,4],[3,4,5,6],[5,6,7,8],[7,8,9,10],
            [9,10,11,12],[11,12,13,14],[13,14,15,16],[15,16,17,18],
            [17,18,19,-3],[19,-4,-5]]

# solve for optimal contraction order
cont_order, costs, is_optimal = ncon_solver(tensors,connects,max_kept=None)

# contract network
T = ncon(tensors,connects,cont_order)


# define a network
chi = 4;
chi = 10
chi_p = 10
u = np.random.rand(chi,chi,chi_p,chi_p)
w = np.random.rand(chi_p,chi_p,chi)
ham = np.random.rand(chi,chi,chi,chi,chi,chi)
rho = np.random.rand(chi,chi,chi,chi,chi,chi)
tensors = [u,u,w,w,w,ham,u,u,w,w,w,rho]
connects = [[1,3,10,11],[4,7,12,13],[8,10,18],[11,12,19],[13,14,20],
	[2,5,6,3,4,7],[1,2,9,17],[5,6,16,15],[8,9,23],[17,16,22],[15,14,21],
	[23,22,21,18,19,20]]

# solve for optimal contraction order
cont_order, costs, is_optimal = ncon_solver(tensors,connects,max_kept=None)

# contract network
T = ncon(tensors,connects,cont_order)







