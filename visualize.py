import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# for use with 3D complex wavefunctions in straight array form. 

def cross_section_real(shape: tuple, psi: np.ndarray, index: int = -1, plane: int = 0):
    wavefunction = psi.reshape(shape)
    if index == -1:
        index = shape[plane] // 2
    
    if plane == 0:
        section = wavefunction[index, :, :]
    elif plane == 1:
        section = wavefunction[:, index, :]
    else:
        section = wavefunction[:, :, index]
    
    plt.imshow(section)
    plt.show()

def cross_section_magnitude(shape: tuple, psi: np.ndarray, index: int = -1, plane: int = 0):
    wavefunction = psi.reshape(shape) 
    if index == -1:
        index = shape[plane] // 2
    
    if plane == 0:
        section = wavefunction[index, :, :]
    elif plane == 1:
        section = wavefunction[:, index, :]
    else:
        section = wavefunction[:, :, index]
    
    plt.imshow(np.abs(np.square(section)))
    plt.show()
    
def cross_section_color(shape: tuple, psi: np.ndarray, index: int = -1, plane: int = 0):
    wavefunction = psi.reshape(shape) 
    if index == -1:
        index = shape[plane] // 2
    
    if plane == 0:
        section = wavefunction[index, :, :]
    elif plane == 1:
        section = wavefunction[:, index, :]
    else:
        section = wavefunction[:, :, index]
    
    magnitude = np.abs(section)
    phase = np.angle(section)

    hue = phase / (2 * np.pi) + 0.5
    saturation = np.ones(hue.shape)
    value = magnitude / np.max(magnitude)

    hsv = np.dstack((hue, saturation, value))
    rgb = hsv_to_rgb(hsv)

    plt.imshow(rgb)
    plt.show()

def radial_density_function(shape: tuple, psi:np.ndarray, center: tuple, dx: int = 1):
    wavefunction = psi.reshape(shape)

    probability = np.abs(np.square(wavefunction))

    r = []
    p = []

    for point in np.ndindex(shape):
        r.append(np.linalg.norm(np.array(point) - np.array(center)) * dx)
        p.append(probability[point])

    plt.hist(r, weights=p, bins=shape[0]//2)
    plt.show()
    
if __name__ == "__main__":
    shape = (50, 50, 50)
    for k in range(5):
        psi = np.load(f'wavefunctions/coulomb_k{k}_m100_{shape}.npy')
        cross_section_color(shape, psi, index=25, plane=2)


    
    