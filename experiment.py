import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

import algorithms
import hamiltonian
import visualize
import search

# particle physics natural units are used: c = hbar = e0 = 1
# from fine structure constant alpha = 1/137.036
# elementary charge = sqrt(4 pi alpha) = 0.30282
# mass of electron = 0.51099895069 MeV


class Datapoint:
    def __init__(self, resolution: int, krylov_dimension: int, levels: int):
        # PARAMETERS (INDEPENDENT VARIABLES)
        self.resolution = resolution
        self.krylov_dimension = krylov_dimension
        self.levels = range(levels)

        self.shape = (resolution, resolution, resolution) # cube with side length of resolution
        self.middle = resolution // 2 # XYZ coordinate of cube center

        self.SCALE = 50 / resolution # in bohr radii. Equal to 50/resolution so that the cube has constant side length 50 bohr radii.
        self.GRID_SPACE = 268 * self.SCALE # Mev-1. The bohr radius, the peak of 1s, should be at 268 Mev-1 or 1/(Me alpha). 

        # PHYSICAL CONSTANTS
        self.ELECTRON_MASS = 0.51099895069 # MeV
        self.ELECTRON_CHARGE = 0.30282 # dimensionless in energy units. 
        
        # HAMILTONIAN
        self.V = hamiltonian.coulomb_potential(self.shape, self.ELECTRON_CHARGE, self.ELECTRON_CHARGE, 
            (self.middle + 0.5, self.middle + 0.5, self.middle + 0.5), self.GRID_SPACE) # nucleus offset by 0.5 to avoid pole at r=0
        self.H = hamiltonian.build_hamiltonian(self.shape, self.V, self.ELECTRON_MASS, self.GRID_SPACE)
        
        # EIGENVALUES AND EIGENVECTORS
        self.eigenvalues, self.eigenvectors = search.search_v1(self.H, m=self.krylov_dimension, k_list=self.levels, verbose=False)

    def plot_eigenvector(self, k: int):
        visualize.cross_section_color(self.shape, self.eigenvectors[k], index=self.middle, plane=2)

    def plot_radial_density(self, k: int):
        visualize.radial_density_function(self.shape, self.eigenvectors[k], center=(self.middle + 0.5, self.middle + 0.5, self.middle + 0.5), dx=self.GRID_SPACE)
        
    def eigenvector_error(self, k: int):
        error = np.linalg.norm(self.H @ self.eigenvectors[k] - self.eigenvalues[k] * self.eigenvectors[k])
        return error

    def energy_mse(self, plot: bool = False):
        ground_state = -0.03125 * self.ELECTRON_MASS * self.ELECTRON_CHARGE**4 / (np.pi**2) # from the analytical solution to the schrodinger equation with the natural units. Should equal Me e^4 / 32 pi^2 hbar^2. 
        exact_eigenvalues = [ground_state / n**2 for n in range(1, len(self.levels)+1)] # these should be E1 times n^-2 from the analytical solution. 
        
        if plot:
            plt.plot(range(1, len(self.levels)+1), self.eigenvalues, label='calculated')
            plt.plot(range(1, len(self.levels)+1), exact_eigenvalues, label='exact')
            plt.legend()
            plt.show()

        errors = self.eigenvalues - exact_eigenvalues
        mse = np.dot(errors, errors)
        return mse

def run_experiment(resolutions: list, krylov_dimensions: list, levels: int):
    r = len(resolutions)
    k = len(krylov_dimensions)
    energy_levels = np.zeros((r, k, levels)) # energy_levels[i,j,k] is the ith resolution, jth krylov dimension, kth energy level.
    mean_squared_errors = np.zeros((r, k)) # mean_squared_errors[i,j] is the mse compared to analytical solution of the ith resolution, jth krylov dimension.

    for i in range(r):
        for j in range(k):
            Test = Datapoint(resolution=resolutions[i], krylov_dimension=krylov_dimensions[j], levels=levels)
            energy_levels[i, j] = Test.eigenvalues
            mean_squared_errors[i, j] = Test.energy_mse()
            del Test

    plt.imshow(np.log10(mean_squared_errors), cmap='viridis')
    plt.colorbar(label='mean squared error (orders of magnitude)')
    plt.xlabel('krylov dimension')
    plt.ylabel('resolution')
    plt.show()

run_experiment(resolutions=[20, 40, 60, 80, 100], krylov_dimensions=[20, 40, 60, 80, 100], levels=5)
