import numpy as np
from schroedinger import *
from plot import *

test = d1schroedinger(F=1, L=7)

E = 6.5+4j
N = 50
n = 20

def calcNorms(E, N, n):
    x = np.linspace(0, test.L, N)
    dx = x[1] - x[0]
    y = np.linspace(0, test.L, N)
    x, y = np.meshgrid(x, y, indexing="ij")
    G01 = test.G0(x, y, E)
    G02 = test.G0(x, y, E - test.F*test.L/2)
    VG01 = test.F * x * G01 * dx
    VG02 = (test.F * x - test.F*test.L/2) * G02 * dx
    realG = test.G(x, y, E)
    G1 = G01
    G2 = G02
    G1norms = np.array([])
    G2norms = np.array([])
    for i in range(n):
        G1 = G01 + G1 @ VG01
        G2 = G02 + G2 @ VG02
        G1norms = np.append(G1norms, dx * np.linalg.norm(G1-realG, 2))
        G2norms = np.append(G2norms, dx * np.linalg.norm(G2-realG, 2))
    return G1norms, G2norms

norm1, norm2 = calcNorms(E, 50, 50)
plt.figure(figsize=(4*1.2,3*1.2))
plt.plot(np.log(norm1), ".", label="$N=50$ eredeti")
plt.plot(np.log(norm2), ".", label="$N=50$ eltolt")
norm1, norm2 = calcNorms(E, 100, 50)
plt.plot(np.log(norm1), ".", label="$N=100$ eredeti")
plt.plot(np.log(norm2), ".", label="$N=100$ eltolt")
norm1, norm2 = calcNorms(E, 200, 50)
plt.plot(np.log(norm1), ".", label="$N=200$ eredeti")
plt.plot(np.log(norm2), ".", label="$N=200$ eltolt")
plt.grid()
plt.legend()
plt.xlabel("$n$")
plt.ylabel("$\\log\\left(||\\hat{G}_n-\\hat{G}||/b\\right)$")
plt.savefig("../figs/convergencerate.pdf")
plt.show()