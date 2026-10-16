import numpy as np
from scipy import sparse
# H = -hbar**2 / (2*m) * Laplacian + V
# particle physics natural units are used: c = hbar = e0 = 1
def index(shape: tuple, coord: tuple): # Convert from x, y, z coordinates to vector indexing. Each grid point is a vector element.
    idx = 0
    for i in range(len(shape)):
        idx *= shape[i]
        idx += coord[i]
    return idx
    
def build_hamiltonian(shape: tuple, V: np.ndarray, m: float, dx: float):
    # The element (a,b) of this matrix is the effect of the bth element of x on the ath element of Hx.
    row = [] 
    col = []
    obj = []
    dimensions = len(shape)
    # V in the Hamiltonian is element-wise multiplied to the wave function. The jth element of Vx only depends on the jth element of x, so V is a diagonal matrix with V(j,j) equal to the potential energy at point j in the grid.
    # T uses the second finite differences approximation. See Equations (17) and (18). The jth element of Tx only depends on the adjacent grid points. The jth element of x contributes -2 per unit to Tx while adjacent elements contribute 1 per unit as shown in Equation (17). This means that in the matrix T there must be a 1 at every (j,k) where k is next to j on the grid, and a -2 at every (j,j) for each dimension, totaling -2 * dims at (j,j). Then multiply by constant -hbar**2/(2*m*dx**2).
    for point in np.ndindex(shape): # For every point (x,y,z).
        # add the diagonal part of T and V, which is -6 * -hbar**2/(2*m*dx**2) and U(x,y,z) respectively.
        idx = index(shape, point)
        row.append(idx)
        col.append(idx) 
        obj.append(-0.5 / (m * dx**2) * (-2 * dimensions) + V[idx]) 
        # find adjacent points in each dimension x, y, and z.
        for dim in range(len(shape)):      
            if point[dim] > 0: # if an adjacent grid point exists on the right/top/back,
                row.append(idx) # row: current point index
                col.append(index(shape, np.array(point) - np.eye(dimensions, dtype=int)[dim])) # column: adjacent point index
                obj.append(-0.5 / (m * dx**2)) # set value to -hbar**2/(2*m*dx**2).
            if point[dim] < shape[dim] - 1: # if an adjacent grid point exists on the left/bottom/front,
                row.append(idx) # row: current point index
                col.append(index(shape, np.array(point) + np.eye(dimensions, dtype=int)[dim])) # column: adjacent point index
                obj.append(-0.5 / (m * dx**2)) # set value to -hbar**2/(2*m*dx**2).
    H = sparse.csr_matrix((obj, (row, col)), shape=(np.prod(shape), np.prod(shape))) # build the sparse matrix
    return H
    
### DIFFERENT POTENTIALS

def coulomb_potential(shape: tuple, charge1: float, charge2: float, nucleus_pos: tuple, dx: float):
    V = np.zeros(np.prod(shape))
    reciprocalk = 0.25 / np.pi # 1/4pi (epsilon0 is 1 in natural units)
    charge = charge1 * charge2 * reciprocalk

    for point in np.ndindex(shape):
        idx = index(shape, point)
        # Distance r = |point - nucleus_pos| * dx
        r = np.linalg.norm(np.array(point) - np.array(nucleus_pos)) * dx
        V[idx] = -charge / r

    return V



    
    