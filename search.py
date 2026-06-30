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
BOHR_RADIUS = 268 # Mev-1. The bohr radius, the peak of 1s, should be at 268 Mev-1 or 1/(Me alpha). 

def search_v1(H: sparse.csr_matrix, m: int, k_list: list, verbose=False): # general tool for H
    n = H.shape[0]
    v0 = np.ones(n, dtype=np.complex128)
    Q, alphas, betas = algorithms.lanczos(H, v0, m)
    if verbose: print(f'lanczos completed with {m} dimensions')

    T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    ritzvalues, ritzvectors = np.linalg.eigh(T)
    if verbose: print(f'ritz vectors and values found')
    
    eigenvectors = []
    for k in k_list:
        eigenvectors.append(Q @ ritzvectors[:, k])

    return ritzvalues[k_list], eigenvectors

def simulate_hydrogen(delta: int, resolution: int, krylov_dimension: int, levels: int, 
                        verbose=False, save_eigenvectors=False, show_eigenvectors=False):
    # specific tool for hydrogen atom
    # mostly calling algorithms.py and hamiltonian.py
    # total time complexity O(r^3 m^2)
    # all unit based constants defined at top of file.
    if verbose: print(f'starting simulation: delta={delta}, resolution={resolution}, krylov_dimension={krylov_dimension}, levels={levels}')
    
    # define cubic grid. O(1)
    shape = (resolution, resolution, resolution)
    middle = resolution // 2
    levels_list = np.arange(levels)
    N = resolution**3
    if verbose: print(f'grid shape generated with {N} points')
    
    # define coulomb potential: (N, N) diagonal matrix. nucleus offset between 8 grid points. O(r^3)
    V = hamiltonian.coulomb_potential(shape, ELECTRON_CHARGE, ELECTRON_CHARGE, (middle + 0.5, middle + 0.5, middle + 0.5), delta)
    if verbose: print(f'coulomb potential defined')
    
    # build hamiltonian: (N, N) sparse matrix. high coefficient constructing laplacian 7 non-zero elements per row. O(r^3)
    H = hamiltonian.build_hamiltonian(shape, V, ELECTRON_MASS, delta)
    if verbose: print(f'hamiltonian built with {H.nnz} entries')

    # run lanczos: uniform initial guess. alphas diagonals, betas off-diagonals of tridiagonal T: (m, m). Q orthonormal: (N, m). with reorthogonalization, O(r^3 m^2)
    v0 = np.ones(N, dtype=np.complex128)
    Q, alphas, betas = algorithms.lanczos(H, v0, krylov_dimension)
    if verbose: print(f'lanczos completed with {krylov_dimension} dimensions')
    
    # diagonalize T: (m, m) matrix. not tridiagonal-specifc for m^2 time, but not bottleneck. O(m^3)
    T = np.diag(alphas.real) + np.diag(betas.real, 1) + np.diag(betas.real, -1)
    ritzvalues, ritzvectors = np.linalg.eigh(T)
    if verbose: print(f'ritz vectors and values found')
    
    # construct eigenvectors: (N) by applying Q to ritzvectors. O(r^3 k)
    eigenvectors = []
    for k in levels_list:
        eigenvectors.append(Q @ ritzvectors[:, k])
    if verbose: print(f'eigenvectors computed')

    if save_eigenvectors:
        for k, eigenvector in enumerate(eigenvectors):
            np.save(f'wavefunctions/coulomb_k{k}_m{krylov_dimension}_E({ritzvalues[k]:.3g})_S{shape}_dx{delta}.npy', eigenvector)
        if verbose: print(f'eigenvectors saved')

    if show_eigenvectors:
        for k, eigenvector in enumerate(eigenvectors):
            visualize.cross_section_color(shape, eigenvector, index=middle, plane=2)
        if verbose: print(f'eigenvectors shown')
    
    return ritzvalues[levels_list], eigenvectors

if __name__ == "__main__":
    simulate_hydrogen(delta=0.4*BOHR_RADIUS, resolution=150, krylov_dimension=100, levels=4, verbose=True, save_eigenvectors=False, show_eigenvectors=True)


    
    

    
    