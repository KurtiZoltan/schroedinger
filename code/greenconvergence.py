import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from schroedinger import *
from plot import *

test = d1schroedinger(F=0.1)

Emin = 0.01
Emax = 4.0
Eimag = 2.0
width = 300
n = 10
N = 100

Ereal = np.linspace(Emin, Emax, width)
height = int(Eimag / (Emax - Emin) * width)
Eimag = np.linspace(0, Eimag, height)#lol
Ereal, Eimag = np.meshgrid(Ereal, Eimag)
E = Ereal + 1j*Eimag
conv = np.empty(E.shape, dtype=float)
x = np.linspace(0, test.L, N)
y = np.linspace(0, test.L, N)
x, y = np.meshgrid(x, y, indexing="ij")

def convergence(E):
    G0 = test.G0(x, y, E)
    VG0 = test.F * x * G0 / N * test.L
    G = G0
    norm = 0
    exponentsum = 0
    for i in range(n):
        Gprev = G
        G = G0 - G @ VG0
        diff = G - Gprev
        prev = norm
        norm = np.sum(np.abs(diff)**2)
        exponentsum += prev / norm / (n - 1)
    if exponentsum < 1:
        exponentsum = 0
    return exponentsum

for i in range(E.shape[0]):
    print(f"{i / E.shape[0] * 100:.1f}%", end="\r")
    for j in range(E.shape[1]):
        conv[i, j] = convergence(E[i, j])

plt.imshow(conv)
plt.show()