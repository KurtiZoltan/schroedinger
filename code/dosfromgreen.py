import numpy as np
from schroedinger import *
from plot import *

N = 1000
epsilon = 1e-2

test = d1schroedinger()
E = np.linspace(0, 14, 10000)
rho1 = np.zeros((0))
rho2 = np.zeros((0))
x = np.linspace(0, test.L, N)
x = x[1:-1]
dx = test.L / (N - 1)
for currE in E:
    temp = 1 / np.pi * np.imag(np.sum(test.G(x, x, currE + 1j * epsilon)) * dx)
    rho1 = np.append(rho1, temp)
    temp = 1 / np.pi * np.imag(np.sum(test.G(x, x, currE + 10j * epsilon)) * dx)
    rho2 = np.append(rho2, temp)
plt.plot(np.real(E), rho1)
plt.plot(np.real(E), rho2)
plt.grid()
plt.savefig("../figs/dosfromgreen.pdf")
plt.show()

N1 = np.zeros((0))
N2 = np.zeros((0))
for i in range(len(E)):
    N1 = np.append(N1, np.sum(rho1[0:i]) * (E[1] - E[0]))
    N2 = np.append(N2, np.sum(rho2[0:i]) * (E[1] - E[0]))
test.eLevel(15)
elevels = test.Es
y = np.zeros(E.shape)
for i, currentx in zip(range(len(E)), E):
    y[i] = np.sum(elevels < currentx)
plt.plot(E, N1)
plt.plot(E, N2)
plt.plot(E, y, alpha=0.6)
plt.grid()
plt.savefig("../figs/numberofstatesfromgreen.pdf")
plt.show()