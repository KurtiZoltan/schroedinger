import numpy as np
from schroedinger import *
from plot import *

test = d1schroedinger(F=1, L=7)

E = 6.5+3j
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
        np.append(G1norms, dx * np.linalg.norm(G1-realG, 2))
        np.append(G2norms, dx * np.linalg.norm(G2-realG, 2))
    return G1norms, G2norms