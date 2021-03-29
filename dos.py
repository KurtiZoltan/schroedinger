import numpy as np
from schroedinger import *
from plot import *

def semiclassical(E):
    F = test.F
    L = test.L
    n1 = 1/np.pi * 2 / 3 / F * (np.power(E, 3/2))
    n2 = 1/np.pi * 2 / 3 / F * (np.power(E, 3/2) - np.power(np.maximum(E - F*L, 0), 3/2))
    n1mask = E < L*F
    return n1 * n1mask + n2 * (1-n1mask)

test = d1schroedinger(F = 1)
test.eLevel(15)
elevels = test.Es
x = np.linspace(0, elevels[-1], 10000)
y = np.zeros(x.shape)
for i, currentx in zip(range(len(x)), x):
    y[i] = np.sum(elevels < currentx)

plt.figure(figsize=[5, 4])
plt.plot(x, y, label = "Állapotszám")
plt.plot(x, semiclassical(x), label = "Szemiklasszikus közelítés")
plt.xlabel("$E$")
plt.ylabel("$\\sum_n \\theta\\left(E - E_n\\right)$")
plt.grid()
plt.legend()
plt.savefig("figs/alapotszam.pdf")
plt.show()