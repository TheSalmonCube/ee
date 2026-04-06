import numpy as np

from scipy import sparse

# H = h/2m ∂x^2 + V

# natural units are used, h bar is 1

# 3 dimensions

def index(shape: tuple, coord: tuple):
    idx = 0
    for i in range(len(shape)):
        idx *= shape[i]
        idx += coord[i]
    return idx
    
def build_hamiltonian(shape: tuple, V: np.ndarray, m: float, dx: float):
    row = []
    col = []
    obj = []
    dimensions = len(shape)
    
    # This hamiltonian is sparse N by N where N is the number of points in the grid. 
    # I treat entire grid as a vector, rather than a dimensional tensor, to avoid multi-dim tensor math.
    # If x is in the jth row and kth column of the hamiltonian means that the jth point in the grid of position space
    # contributes x to the kth point in the grid in energy space.

    # V in the Hamiltonian is element-wise multiplied to the wave function. This is represented by a diagonal matrix,
    # with V(j,j) the potential energy at point j in the grid.

    # K is more complicated. I decided to use the second finite difference for the second partial derivative (Laplacian)
    # In one dimension, it is applying the kernel [ 1 -2 1 ] to each grid point, yielding a tridiagonal matrix.
    # Generalizing to n dimensions, there must be a 1 at every (j,k) where k is next to j on the grid.
    # and a -2 at every (j,j) for each dimension, totaling -2 * dims at (j,j). Then multiply by constant -h/2m.

    for point in np.ndindex(shape):
        # DIAGONAL
        # Kinetic: -1/2m (-2 * dimensions) / dx^2
        # Potential: V(point)

        idx = index(shape, point)
        row.append(idx)
        col.append(idx)
        obj.append(-0.5 / m * (-2 * dimensions) + V[idx])

        # OFF DIAGONAL
        # Kinetic: 1/2m * (1) / dx^2 for each dimension

        for dim in range(len(shape)):      
            if point[dim] > 0:
                row.append(idx)
                col.append(index(shape, np.array(point) - np.eye(dimensions, dtype=int)[dim])) # np.eye(dimensions)[dim] is [0, ... 1 ... 0] with 1 on dim, so that when subtracted from point i get the point below it on that dimension
                obj.append(-0.5 / (m * dx**2))
            
            if point[dim] < shape[dim] - 1:
                row.append(idx)
                col.append(index(shape, np.array(point) + np.eye(dimensions, dtype=int)[dim])) # np.eye(dimensions)[dim] is [0, ... 1 ... 0] with 1 on dim, so that when added to point i get the point above it on that dimension
                obj.append(-0.5 / (m * dx**2))
        
    # BUILD
    H = sparse.csr_matrix((obj, (row, col)), shape=(np.prod(shape), np.prod(shape)))
    return H
    
### DIFFERENT POTENTIALS

def coulomb_potential(shape: tuple, charge: float, nucleus_pos: tuple):
    V = np.zeros(np.prod(shape))

    for point in np.ndindex(shape):
        idx = index(shape, point)
        V[idx] = -charge / np.linalg.norm(np.array(point) - np.array(nucleus_pos))

    return V


if __name__ == "__main__":
    shape = (50, 50, 50)
    V = coulomb_potential(shape, 1, (25.5, 25.5, 25.5))
    H = build_hamiltonian(shape, V, 1, 1)
    print(H.nnz)
    
    