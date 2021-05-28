import numpy as np
from scipy.optimize import curve_fit
from schroedinger import *
from plot import *

test = d1schroedinger(L=7)

# Emin = 0.1
# Emax = 10
# Eimagmax = 1.1
# Eimagmin = -1.1
# width = 1000
# N = 50 # N^2 number of points in discretization
# numEnergies = 20

Emin = 0.1 * np.power(10, 2/3)
Emax = 2 * np.power(10, 2/3)
Eimagmax = 1.1 * np.power(10, 2/3)
Eimagmin = -0.5 * np.power(10, 2/3)
width = 500
N = 50
#n = 10
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
dx = x[1] - x[0]
y = np.linspace(0, test.L, N)
x, y = np.meshgrid(x, y, indexing="ij")

def normguess(steps, alpha, remainder):
    return np.exp(steps*alpha) + remainder

def convergence(E):
    G0 = test.G0(x, y, E)
    VG0 = test.F * x * G0 / N * test.L
    realG = test.G(x, y, E)
    G = G0
    norm0 = dx * np.linalg.norm(G0 @ VG0, ord=2)
    norms = np.array([norm0])
    steps = np.array([0])
    for i in range(20):
        Gprev = G
        G = G0 + G @ VG0
        norm = dx * np.linalg.norm(G - realG, ord=2)
        norms = np.append(norms, norm)
        steps = np.append(steps, i+1)
        if norm/norm0 + norm0/norm > 5:
            break
    
    popt, pcov = curve_fit(normguess, steps, norms/norms[0])
    return -popt[0]

def convergenceImproved(E):
    G0 = test.G0(x, y, E - test.F * test.L / 2)
    VG0 = (test.F * x - test.F * test.L / 2) * G0 / N * test.L
    realG = test.G(x, y, E)
    G = G0
    norm0 = dx * np.linalg.norm(G0 @ VG0, ord=2)
    norms = np.array([norm0])
    steps = np.array([0])
    for i in range(20):
        Gprev = G
        G = G0 + G @ VG0
        norm = dx * np.linalg.norm(G - realG, ord=2)
        norms = np.append(norms, norm)
        steps = np.append(steps, i+1)
        if norm/norm0 + norm0/norm > 5:
            break
    
    popt, pcov = curve_fit(normguess, steps, norms/norms[0])
    return -popt[0]

# for i in range(E.shape[0]):
#     print(f"{i / E.shape[0] * 100:.1f}%", end="\r")
#     for j in range(E.shape[1]):
#         conv[i, j] = convergence(E[i, j])
#         convImproved[i, j] = convergenceImproved(E[i, j])

#np.savetxt("../cache/convImproved.txt", convImproved)
#np.savetxt("../cache/conv.txt", conv)
conv = np.loadtxt("../cache/conv300.txt")
convImproved = np.loadtxt("../cache/convImproved300.txt")

conv[conv < 0] = -np.max(conv)/10
#conv[conv <= 0] = (conv[conv >= 1] - 1) / (np.max(conv[conv >= 1]) - 1) + 1
convImproved[convImproved < 0] = -np.max(convImproved)/10
#convImproved[convImproved >= 1] = (convImproved[convImproved >= 1] - 1) / (np.max(convImproved[convImproved >= 1]) - 1) + 1

cutoff = 9
if cutoff > 0:
    conv = conv[:-cutoff]
    convImproved = convImproved[:-cutoff]
    Eimagmax = np.max(Eimag[:-cutoff])

# phi = np.linspace(0, 2*np.pi, 3000)
# fig, axs = plt.subplots(2, 1)
# im1 = axs[0].imshow(conv, origin="lower", aspect="equal", extent=(Emin, Emax, Eimagmin, Eimagmax), cmap=plt.get_cmap("magma"))
# fig.colorbar(im1, ax=axs[0])
# im2 = axs[1].imshow(convImproved, origin="lower", aspect="equal", extent=(Emin, Emax, Eimagmin, Eimagmax),cmap=plt.get_cmap("magma"))
# fig.colorbar(im2, ax=axs[1])
# r = test.F * test.L / 2
# for E, realE in zip(Es, test.Es):
#     if Emin < E + test.F * test.L / 2 and E + test.F * test.L / 2 < Emax:
#         plt.plot(E + test.F * test.L / 2, 0, "rx", alpha=1)
#     if Emin < realE and realE < Emax:
#         plt.plot(realE, 0, "yx", alpha=1)
#     x = r * np.cos(phi) + E + test.F * test.L / 2
#     y = r * np.sin(phi)
#     xinside = x[(Emin < x) * (x < Emax) * (Eimagmin < y) * (y < Eimagmax)]
#     yinside = y[(Emin < x) * (x < Emax) * (Eimagmin < y) * (y < Eimagmax)]
#     axs[1].plot(xinside, yinside, "r", alpha=0.4)
# axs[0].set_xlabel("$\\mathrm{Re}(bE)$")
# axs[0].set_ylabel("$\\mathrm{Im}(bE)$")
# axs[1].set_xlabel("$\\mathrm{Re}(bE)$")
# axs[1].set_ylabel("$\\mathrm{Im}(bE)$")
# plt.savefig("../figs/convergence4.pdf")
# plt.show()

plt.figure(figsize=(4,3))
plt.imshow(conv, origin="lower", aspect="equal", extent=(Emin, Emax, Eimagmin, Eimagmax), cmap=plt.get_cmap("magma"))
plt.xlabel("$\\mathrm{Re}(bE)$")
plt.ylabel("$\\mathrm{Im}(bE)$")
plt.colorbar()
plt.savefig("../figs/convergenceOriginal.pdf")
plt.show()

plt.figure(figsize=(4,3))
plt.imshow(convImproved, origin="lower", aspect="equal", extent=(Emin, Emax, Eimagmin, Eimagmax), cmap=plt.get_cmap("magma"))
r = test.F * test.L / 2
phi = np.linspace(0, 2*np.pi, 3000)
for E, realE in zip(Es, test.Es):
    if Emin < E + test.F * test.L / 2 and E + test.F * test.L / 2 < Emax:
        plt.plot(E + test.F * test.L / 2, 0, "rx", alpha=1)
    if Emin < realE and realE < Emax:
        plt.plot(realE, 0, "yx", alpha=1)
    x = r * np.cos(phi) + E + test.F * test.L / 2
    y = r * np.sin(phi)
    xinside = x[(Emin < x) * (x < Emax) * (Eimagmin < y) * (y < Eimagmax)]
    yinside = y[(Emin < x) * (x < Emax) * (Eimagmin < y) * (y < Eimagmax)]
    plt.plot(xinside, yinside, "r", alpha=0.4)
plt.xlabel("$\\mathrm{Re}(bE)$")
plt.ylabel("$\\mathrm{Im}(bE)$")
plt.colorbar()
plt.savefig("../figs/convergenceImproved.pdf")
plt.show()