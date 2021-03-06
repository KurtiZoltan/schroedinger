import numpy as np
from schroedinger import *
from plot import *

test = d1schroedinger(F=0.1*10, L=15/np.power(10,1/3))

E = 6.5+4j
n = 50

N = 1000
x = np.linspace(0, test.L, N)
y = np.linspace(0, test.L, N)
x, y = np.meshgrid(x, y, indexing="ij")
G01 = test.G0(x, y, E)
G02 = test.G0(x, y, E - test.F*test.L/2)
VG01 = test.F * x * G01 / (N-2) * test.L
VG02 = (test.F * x - test.F*test.L/2) * G02 / (N-2) * test.L

G1 = G01
G2 = G02
plt.imshow(np.abs(G2), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.title("free G2")
plt.show()
for i in range(n):
    Gprev1 = G1
    Gprev2 = G2
    G1 = G01 + G1 @ VG01
    G2 = G02 + G2 @ VG02
    if i % 20 == 0:
        plt.imshow(np.abs(G2 - Gprev2), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
        plt.colorbar()
        plt.title(str(i))
        plt.show()


plt.imshow(np.abs(G2), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
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
plt.imshow(np.abs(realG - G2), aspect="equal", origin="lower", extent=(0, test.L, 0, test.L))
plt.colorbar()
plt.title("error")
plt.show()

