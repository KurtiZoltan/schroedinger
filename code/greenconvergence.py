import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from schroedinger import *
from plot import *

test = d1schroedinger(F=0.1)

# Emin = 0.1
# Emax = 10
# Eimagmax = 1.1
# Eimagmin = -1.1
# width = 1000
# N = 50 # N^2 number of points in discretization
# numEnergies = 20

Emin = 0.1
Emax = 2
Eimagmax = 1.1
Eimagmin = -0.5
width = 1000
N = 50
n = 10
numEnergies = 10

Es = np.pi**2 / test.L**2 * np.arange(1, numEnergies+1) ** 2
test.eLevel(numEnergies-1)
Ereal = np.linspace(Emin, Emax, width)
height = int((Eimagmax - Eimagmin) / (Emax - Emin) * width)
Eimag = np.linspace(Eimagmin, Eimagmax, height)
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
    norm0 = np.linalg.norm(G0 @ VG0, ord=2)
    iterations = 0
    while True:
        iterations += 1
        Gprev = G
        G = G0 - G @ VG0
        norm1 = np.linalg.norm(G - Gprev, ord=2)
        if norm1 / norm0 < 1 / 3:
            return iterations
        if norm1 / norm0 > 3:
            return 0
        if iterations == 50:
            return 0

def convergenceImproved(E):
    G0 = test.G0(x, y, E - test.F * test.L / 2)
    VG0 = (test.F * x - test.F * test.L / 2) * G0 / N * test.L
    G = G0
    norm0 = np.linalg.norm(G0 @ VG0, ord=2)
    iterations = 0
    while True:
        iterations += 1
        Gprev = G
        G = G0 - G @ VG0
        norm1 = np.linalg.norm(G - Gprev, ord=2)
        if norm1 / norm0 < 1 / 3:
            return iterations
        if norm1 / norm0 > 3:
            return 0
        if iterations == 50:
            return 0
'''
for i in range(E.shape[0]):
    print(f"{i / E.shape[0] * 100:.1f}%", end="\r")
    for j in range(E.shape[1]):
        conv[i, j] = convergence(E[i, j])
        convImproved[i, j] = convergenceImproved(E[i, j])
'''
#np.savetxt("../cache/convImproved4.txt", convImproved)
#np.savetxt("../cache/conv4.txt", conv)
conv = np.loadtxt("../cache/conv4.txt")
convImproved = np.loadtxt("../cache/convImproved4.txt")

# conv[conv < 1] = 0
# conv[conv >= 1] = (conv[conv >= 1] - 1) / (np.max(conv[conv >= 1]) - 1) + 1
# convImproved[convImproved < 1] = 0
# convImproved[convImproved >= 1] = (convImproved[convImproved >= 1] - 1) / (np.max(convImproved[convImproved >= 1]) - 1) + 1

phi = np.linspace(0, 2*np.pi, 3000)
fig, axs = plt.subplots(2, 1)
im1 = axs[0].imshow(conv, origin="lower", aspect="equal", extent=(Emin, Emax, Eimagmin, Eimagmax), cmap=plt.get_cmap("magma"))
fig.colorbar(im1, ax=axs[0])
im2 = axs[1].imshow(convImproved, origin="lower", aspect="equal", extent=(Emin, Emax, Eimagmin, Eimagmax),cmap=plt.get_cmap("magma"))
fig.colorbar(im2, ax=axs[1])
r = test.F * test.L / 2
for E, realE in zip(Es, test.Es):
    if Emin < E + test.F * test.L / 2 and E + test.F * test.L / 2 < Emax:
        plt.plot(E + test.F * test.L / 2, 0, "rx", alpha=1)
    if Emin < realE and realE < Emax:
        plt.plot(realE, 0, "yx", alpha=1)
    x = r * np.cos(phi) + E + test.F * test.L / 2
    y = r * np.sin(phi)
    xinside = x[(Emin < x) * (x < Emax) * (Eimagmin < y) * (y < Eimagmax)]
    yinside = y[(Emin < x) * (x < Emax) * (Eimagmin < y) * (y < Eimagmax)]
    axs[1].plot(xinside, yinside, "r", alpha=0.4)
plt.show()