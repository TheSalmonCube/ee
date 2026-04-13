import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

import algorithms
import hamiltonian
import visualize

# particle physics natural units are used: c = hbar = e0 = 1
# from fine structure constant alpha = 1/137.036
# elementary charge = sqrt(4 pi alpha) = 0.30282
# mass of electron = 0.51099895069 MeV

ELECTRON_MASS = 0.51099895069 # MeV
ELECTRON_CHARGE = 0.30282 # dimensionless
GRID_SPACE = 268 # Mev-1. The bohr radius, the peak of 1s, should be at 268 Mev-1 or 1/(Me alpha).

def search_v1(H: sparse.csr_matrix, m: int, k_list: list):
    n = H.shape[0]
    v0 = np.random.rand(n)
    Q, alphas, betas = algorithms.lanczos(H, v0, m)
    T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    ritzvalues, ritzvectors = np.linalg.eigh(T)
    
    eigenvectors = []
    for k in k_list:
        eigenvectors.append(Q @ ritzvectors[:, k])

    return ritzvalues[k_list], eigenvectors


if __name__ == "__main__":
    shape = (50, 50, 50)
    K = [0, 1, 2, 3, 4]
    M = 100

    V = hamiltonian.coulomb_potential(shape, ELECTRON_CHARGE, ELECTRON_CHARGE, (25.5, 25.5, 25.5), GRID_SPACE)
    H = hamiltonian.build_hamiltonian(shape, V, ELECTRON_MASS, GRID_SPACE)
    print(H.nnz)

    eigenvalues, eigenvectors = search_v1(H, m=M, k_list=K)

    plt.imshow(V.reshape(shape)[:, :, 25])
    plt.show()

    for k, eigenvector in enumerate(eigenvectors):
        np.save(f'wavefunctions/coulomb_k{k}_m{M}_{shape}.npy', eigenvector)
        visualize.cross_section_color(shape, eigenvector, index=25, plane=2)



    
    

    
    