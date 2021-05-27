import numpy as np
from schroedinger import *
from plot import *

N = 100

test = d1schroedinger()
E = np.linspace(0, 7, 5000)
rho1 = np.zeros((0))
rho2 = np.zeros((0))
x = np.linspace(0, test.L, N)
x = x[1:-1]
dx = x[1]-x[0]

for currE in E:
    temp = -1 / np.pi * np.imag(np.sum(test.G(x, x, currE + 1e-2j)) * dx)
    rho1 = np.append(rho1, temp)
    temp = -1 / np.pi * np.imag(np.sum(test.G(x, x, currE + 1e-1j)) * dx)
    rho2 = np.append(rho2, temp)
plt.figure(figsize=(4,3))
plt.plot(np.real(E), rho2, label="$b\\epsilon=0.1i$")
plt.plot(np.real(E), rho1, label="$b\\epsilon=0.01i$")
plt.grid()
plt.legend()
plt.xlabel("$bE$")
plt.ylabel("$\\rho(E)/b$")
plt.savefig("../figs/dosfromgreen.pdf")
plt.show()

N1 = np.zeros((0))
N2 = np.zeros((0))
for i in range(len(E)):
    N1 = np.append(N1, np.sum(rho1[0:i]) * (E[1] - E[0]))
    N2 = np.append(N2, np.sum(rho2[0:i]) * (E[1] - E[0]))
test.eLevel(3)
elevels = test.Es
y = np.zeros(E.shape)
for i, currentx in zip(range(len(E)), E):
    y[i] = np.sum(elevels < currentx)
plt.figure(figsize=(4,3))
plt.plot(E, N2, label="$b\\epsilon=0.1i$")
plt.plot(E, N1, label="$b\\epsilon=0.01i$")
plt.plot(E, y, alpha=0.75, label="Egzakt állapotszám")
plt.grid()
plt.legend()
plt.xlabel("$bE$")
plt.ylabel("$\\sum_n \\theta\\left(E - E_n\\right)$")
plt.savefig("../figs/numberofstatesfromgreen.pdf")
plt.show()