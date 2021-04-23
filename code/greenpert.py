import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from schroedinger import *
from plot import *

test = d1schroedinger(F=0.1)

E = 2 + 1j
n = 100

N = 1000
x = np.linspace(0, test.L, N)
y = np.linspace(0, test.L, N)
x, y = np.meshgrid(x, y, indexing="ij")
G0 = test.G0(x, y, E)
VG0 = test.F * x * G0 / N * test.L

G = G0
#plt.imshow(np.real(G), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
#plt.colorbar()
#plt.show()
for i in range(n):
    Gprev = G
    G = G0 - G @ VG0
    plt.imshow(np.abs(G - Gprev), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
    if i % 110 == 0:
        plt.colorbar()
        plt.title(str(i))
        plt.show()


plt.imshow(np.abs(G), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.title("perturbed G")
plt.show()

realG = test.G(x, y, E)
plt.imshow(np.abs(realG), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.title("explicit formula")
plt.show()
plt.imshow(np.abs(realG - G), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.title("error")
plt.show()