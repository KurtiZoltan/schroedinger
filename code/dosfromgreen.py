import numpy as np
from schroedinger import *
from plot import *

N = 1000
epsilon = 1e-2

test = d1schroedinger()
E = np.linspace(0, 14, 10000)
rho = np.zeros((0))
x = np.linspace(0, test.L, N)
x = x[1:-1]
dx = test.L / (N - 1)
for currE in E:
    temp = 1 / np.pi * np.imag(np.sum(test.G(x, x, currE + 1j * epsilon)) * dx)
    rho = np.append(rho, temp)
plt.plot(np.real(E), rho)
plt.grid()
plt.savefig("../figs/dosfromgreen.pdf")
plt.show()

N = np.zeros((0))
for i in range(len(E)):
    N = np.append(N, np.sum(rho[0:i]) * (E[1] - E[0]))
plt.plot(E, N)
plt.grid()
plt.savefig("../figs/numberofstatesfromgreen.pdf")
plt.show()