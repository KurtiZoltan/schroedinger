import numpy as np
from scipy import special, optimize
from plot import *

def char_eq(E, L):
    ai1, ai1p, bi1, bi1p = special.airy(-E)
    ai2, ai2p, bi2, bi2p = special.airy(L-E)
    f = bi1*ai2 - ai1*bi2
    fp = -(bi1p*ai2 + bi1*ai2p - (ai1p*bi2 + ai1*bi2p))
    return f, fp

def semiclassic(E, L, n):
    if E > L:
        f = 2/3*(np.power(E, 3/2) - np.power(E - L, 3/2)) - (n + 1) * np.pi
        fp = np.sqrt(E) - np.sqrt(E - L)
    else:
        f = 2 / 3 * np.power(E, 3 / 2) - (n + 3/4) * np.pi
        fp = np.sqrt(E)
    return f, fp

def infsquare(E, L, n):
    f = E - (np.pi * (n + 1) / L)**2
    fp = E / E
    return f, fp

def wave_fun(x, E):
    ai1, ai1p, bi1, bi1p = special.airy(-E)
    ai2, ai2p, bi2, bi2p = special.airy(x - E)
    return bi1 * ai2 - ai1 * bi2

# E = np.linspace(30, 40, 1000)
# x = np.linspace(-0.1, 1.1, 1000)
# f, fp = char_eq(E, L=1)
# E0 = 40
# psi = wave_fun(x, E0)
# plt.plot(E, f)
# plt.plot(x, psi)
# plt.grid()
# plt.show()

def band(n, l0, l1):
    L = np.linspace(l0, l1, 100)
    E0 = np.array([])
    Estart = np.pi**2 / L[0]**2 * (n+1)**2
    dl = L[1] - L[0]
    for l in L:
        temp = (optimize.root_scalar(f=char_eq, args=l, x0=Estart, fprime=True)).root
        Estart = temp * (l/(l+dl))**2
        E0 = np.append(E0, temp)
    plt.plot(L, E0, 'r')
    return

def compare(n, l0, l1, appr):
    band(n, l0, l1)
    L = np.linspace(l0, l1, 3000)
    E0 = np.array([])
    Estart = np.pi ** 2 / L[0] ** 2 * (n+1) ** 2
    dl = L[1] - L[0]
    for l in L:
        temp = (optimize.root_scalar(f=appr, args=(l, n), x0=Estart, fprime=True)).root
        Estart = temp * (l / (l + dl)) ** 2
        E0 = np.append(E0, temp)
    plt.plot(L, E0, 'b')
    return

plt.figure(figsize=[4, 3])
for i in range(13):
    band(i, 0.5, 5)
plt.xlim((0.5, 5))
plt.ylim((0, 80))
plt.xlabel("$aL$")
plt.ylabel("$bE$")
plt.grid()
plt.tight_layout()
plt.savefig("../figs/" + "energiaszintek.pdf")
plt.show()

plt.figure(figsize=[4, 3])
for i in range(0, 6):
    compare(i, 0.5, 5, semiclassic)
plt.xlim((1, 5))
plt.ylim((0, 15))
plt.xlabel("$aL$")
plt.ylabel("$bE$")
plt.grid()
plt.tight_layout()
plt.legend(("Egzakt energia", "Szemiklasszikus közelítés"))
plt.savefig("../figs/" + "energiaszintkozelites.pdf")
plt.show()

plt.figure(figsize=[4, 3])
for i in range(0, 15):
    compare(i, 0.5, 3, infsquare)
plt.xlim((0.5, 3))
plt.ylim((0, 100))
plt.xlabel("$aL$")
plt.ylabel("$bE$")
plt.grid()
plt.tight_layout()
plt.legend(("Egzakt energia", "Végtelen potenciálgödör"))
plt.savefig("../figs/" + "infsquareenergia.pdf")
plt.show()
