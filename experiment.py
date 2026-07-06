from search import BOHR_RADIUS
from search import simulate_hydrogen

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

import algorithms
import hamiltonian
import visualize
import search

# particle physics natural units are used: c = hbar = e0 = 1
# from fine structure constant alpha = 1/137.036
# elementary charge = sqrt(4 pi alpha) = 0.30282
# mass of electron = 510998.95069 eV

def run_experiment(grid_spacings: list, krylov_dimensions: list, energy_levels: int, side_length: float = 60, 
                    max_time: float = 0, metric: str = 'percent', save_eigenvectors: bool = False, verbose: bool = False):
        
        """
        Runs an experiment testing the eigenvalue accuracy of different parameters.

        Args:
            grid_spacings: list of grid spacings to test (in Bohr Radii)
            krylov_dimensions: list of krylov dimensions to test
            energy_levels: number of energy levels to test
            side_length: side length of the cubic grid in Bohr Radii
            max_time: maximum time to run the simulation, stops increasing grid_spacings or krylov_dimension if time exceeded
            metric: metric to use for measuring accuracy
            save_eigenvectors: whether to save the eigenvectors
            verbose: whether to print progress updates
        """

        grid_spacings.sort(reverse=True)
        krylov_dimensions.sort()

        eigenvalues = np.zeros((len(grid_spacings), len(krylov_dimensions), energy_levels))
        accuracies = np.zeros((len(grid_spacings), len(krylov_dimensions)))
        times = np.zeros((len(grid_spacings), len(krylov_dimensions)))

        exact_eigenvalues = np.fromfunction(lambda n: -13.6057/(n+1)**2, (energy_levels,))

        for i, delta in enumerate(grid_spacings):
            for j, m in enumerate(krylov_dimensions):
                resolution = np.ceil(side_length/delta).astype(int)

                start_time = time.time()
                eigenvalues[i, j], _ = simulate_hydrogen(delta * BOHR_RADIUS, # delta is in Bohr Radii. Convert to inverse energy length units 
                resolution, m, energy_levels, False, save_eigenvectors, False)
                
                times[i, j] = time.time() - start_time
                if metric == 'mse':
                    accuracies[i, j] = np.sum(np.square(eigenvalues[i, j] - exact_eigenvalues))
                elif metric == 'percent':
                    accuracies[i, j] = 100 * np.mean(np.abs(eigenvalues[i, j] - exact_eigenvalues) / np.abs(exact_eigenvalues))
                
                if verbose:
                    print(f'delta={delta}, m={m}, accuracy={accuracies[i, j]}, time={times[i, j]}')

                if max_time > 0 and times[i, j] > max_time:
                    break
        
        # plot accuracies

        fig, ax = plt.subplots()
        masked_accuracies = np.ma.masked_equal(accuracies, 0)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='black')
        im = ax.imshow(masked_accuracies, origin='lower', norm=mcolors.LogNorm(), aspect='auto', cmap=cmap)
        
        ax.set_xticks(np.arange(len(krylov_dimensions)))
        ax.set_xticklabels(krylov_dimensions)
        ax.set_yticks(np.arange(len(grid_spacings)))
        ax.set_yticklabels(grid_spacings)
        
        ax.set_xlabel('m (dimensions)')
        ax.set_ylabel('delta (Bohr radii)')
        
        cbar = fig.colorbar(im)
        cbar.set_label(metric)
        
        plt.title(f'Accuracy ({metric})')
        plt.show()


        # plot simulation times

        fig, ax = plt.subplots()
        masked_times = np.ma.masked_equal(times, 0)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='black')
        im = ax.imshow(masked_times, origin='lower', norm=mcolors.LogNorm(), aspect='auto', cmap=cmap)
        
        ax.set_xticks(np.arange(len(krylov_dimensions)))
        ax.set_xticklabels(krylov_dimensions)
        ax.set_yticks(np.arange(len(grid_spacings)))
        ax.set_yticklabels(grid_spacings)
        
        ax.set_xlabel('m (dimensions)')
        ax.set_ylabel('delta (Bohr radii)')
        
        cbar = fig.colorbar(im)
        cbar.set_label('time (s)')
        
        plt.title('Computation Time (s)')
        plt.show()

        return eigenvalues, accuracies, times


if __name__ == "__main__":
    eigs, accs, tims = run_experiment(grid_spacings=[1, 0.8, 0.64, 0.512, 0.4096],
    krylov_dimensions=[20, 40, 80, 120, 160, 200], 
    energy_levels=4,  
    verbose=True, save_eigenvectors=False)    

    print(eigs)
    print(accs)
    print(tims)



    