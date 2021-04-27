import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from schroedinger import *
from plot import *

test = d1schroedinger(F=0.1)

E = 10   + 1j
n = 100

N = 500
x = np.linspace(0, test.L, N)
y = np.linspace(0, test.L, N)
x, y = np.meshgrid(x, y, indexing="ij")
G01 = test.G0(x, y, E)
G02 = test.G0(x, y, E - test.F*test.L/2)
VG01 = test.F * x * G01 / N * test.L
VG02 = (test.F * x - test.F*test.L/2) * G02 / N * test.L

G1 = G01
G2 = G02
plt.imshow(np.real(G1), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.title("free G1")
plt.show()
for i in range(n):
    Gprev1 = G1
    G1 = G01 - G1 @ VG01
    G2 = G02 - G2 @ VG02
    plt.imshow(np.abs(G1 - Gprev1), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
    if i % 33 == 0:
        plt.colorbar()
        plt.title(str(i))
        plt.show()


plt.imshow(np.abs(G1), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.title("perturbed G")
plt.show()

plt.imshow(np.abs(G1 - G2), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.title("perturbed G diff")
plt.show()

realG = test.G(x, y, E)
plt.imshow(np.abs(realG), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.title("explicit formula")
plt.show()
plt.imshow(np.abs(realG - G1), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.title("error")
plt.show()

