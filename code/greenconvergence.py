import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from schroedinger import *
from plot import *

test = d1schroedinger(F=0.1)

Emin = 0.1
Emax = 2
Eimagmax = 1.1
width = 1000
n = 15
N = 50

Ereal = np.linspace(Emin, Emax, width)
height = int(Eimagmax / (Emax - Emin) * width)
Eimag = np.linspace(0, Eimagmax, height)#lol
Ereal, Eimag = np.meshgrid(Ereal, Eimag)
E = Ereal + 1j*Eimag
conv = np.empty(E.shape, dtype=float)
convImproved = np.empty(E.shape, dtype=float)
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
    return exponentsum

def convergenceImproved(E):
    G0 = test.G0(x, y, E - test.F * test.L / 2)
    VG0 = (test.F * x - test.F * test.L / 2) * G0 / N * test.L
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
    return exponentsum

for i in range(E.shape[0]):
    print(f"{i / E.shape[0] * 100:.1f}%", end="\r")
    for j in range(E.shape[1]):
        conv[i, j] = convergence(E[i, j])
        convImproved[i, j] = convergenceImproved(E[i, j])

np.savetxt("../cache/conv2.txt", conv)
np.savetxt("../cache/convImproved2.txt", convImproved)

conv[conv < 1] = 0
conv[conv >= 1] = (conv[conv >= 1] - 1) / (np.max(conv[conv >= 1]) - 1) + 1
convImproved[convImproved < 1] = 0
convImproved[convImproved >= 1] = (convImproved[convImproved >= 1] - 1) / (np.max(convImproved[convImproved >= 1]) - 1) + 1

fig, axs = plt.subplots(2, 1)
im1 = axs[0].imshow(conv, origin="lower", aspect="equal", extent=(Emin, Emax, 0, Eimagmax), cmap=plt.get_cmap("magma"))
#fig.colorbar(im1, ax=axs[0])
im2 = axs[1].imshow(convImproved, origin="lower", aspect="equal", extent=(Emin, Emax, 0, Eimagmax),cmap=plt.get_cmap("magma"))
#fig.colorbar(im2, axs[1])
plt.show()