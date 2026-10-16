import numpy as np
import matplotlib.pyplot as plt

## Figure 1 - Example wavefunction
x = np.arange(-100,100,dtype=np.complex64)
sigma = 10
k = 0.25
packet = 1/(np.sqrt(sigma) * (np.pi ** 0.25)) * np.exp(1j * k * x - x**2/(2 * sigma**2))
plt.plot(x, np.real(packet), label="Real part of ψ")
plt.plot(x, np.imag(packet), label="Imaginary part of ψ")
plt.xlabel("Position")
plt.ylabel("Amplitude")
plt.title("Figure 1 - A complex-valued wavefunction ψ")
plt.legend()
plt.show()

## Figure 2 – Its position probabilities

packet_pdf = abs(packet)**2
plt.plot(x, packet_pdf)
plt.xlabel("Position")
plt.ylabel("Probability Density")
plt.title("Figure 2 - Probability density of ψ, |ψ|^2")
plt.legend()
plt.show()

## Figure 3 - Approximating the wave function

gaussian = np.real(1/(np.sqrt(sigma) * (np.pi ** 0.25)) * np.exp(-x**2/(2 * sigma**2)))
scatterx = [range(-50,51,10)]
scattery = [gaussian[n] for n in range(50,151,10)]
print(scattery)
plt.plot(x, gaussian, label="Example wave function:\nInfinitely many values")
plt.scatter(scatterx, scattery, label="Discrete approximation:\nFinite values", c="red")
plt.xlabel("Position")
plt.ylabel("Amplitude")
plt.title("Figure 3 - Outline of discretization process")
plt.legend(loc="upper right")
plt.show()

## Figure 5 – Momentum Operator Eigenvector

pure_wave = np.exp(0.1j * x)
plt.plot(x, np.real(pure_wave), label="Real part of exp(0.3 i x)")
plt.plot(x, np.imag(pure_wave), label="Imaginary part of exp(0.3 i x)")
plt.xlabel("Position")
plt.ylabel("Amplitude")
plt.title("Figure 5 - Momentum Operator Eigenvector")
plt.legend()
plt.show()
plt.plot()

## Figure 6 – Discretized Grid

x_dim = np.linspace(0, 49, 50)
y_dim = np.linspace(0, 49, 50)
z_dim = np.linspace(0, 49, 50)
X, Y, Z = np.meshgrid(x_dim, y_dim, z_dim)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z, s=0.1, alpha=0.6)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
plt.title("Figure 6 - Discretized Grid For n = 50")
plt.show()
plt.plot()

