import matplotlib.pyplot as plt
import numpy as np
from scipy import special, optimize, integrate
from matplotlib import animation

Fx = 0.0001
Fy = 1
Lx = 10
Ly = 15

x0 = 5
y0 = 3
d = 1
kx = 0
ky = 2

max_energy = 20

animTime = 30
FPS = 30
timeSpeed = 1

name = 'vertical_only_corrected.mp4'
save = False

def char_eq(E, L, F):
    F3sqrt = np.power(F, 1/3)
    ai1, ai1p, bi1, bi1p = special.airy(-E / F3sqrt ** 2)
    ai1p /= F3sqrt ** 2
    bi1p /= F3sqrt ** 2
    ai2, ai2p, bi2, bi2p = special.airy(F3sqrt * L - E / F3sqrt ** 2)
    ai2p /= F3sqrt ** 2
    bi2p /= F3sqrt ** 2
    f = bi1*ai2 - ai1*bi2
    fp = -(bi1p*ai2 + bi1*ai2p - (ai1p*bi2 + ai1*bi2p))
    return f, fp

def d1wave_fun(x, E, F):
    F3sqrt = np.power(F, 1/3)
    ai1, ai1p, bi1, bi1p = special.airy(-E / F3sqrt ** 2)
    ai2, ai2p, bi2, bi2p = special.airy(F3sqrt * x - E / F3sqrt ** 2)
    mask = np.array(E / F3sqrt ** 2 - F3sqrt * x > -10).astype(float)
    return (bi1 * ai2 - ai1 * bi2) * mask

def d1energyLevels(E_max, L, F):
    Es = np.zeros((0))
    i = 1
    while True:
        lstart = 1 / np.power(F, 1/3)
        if L <= lstart:
            llist = np.array([L])
            stepsize = float("nan")
        else:
            stepsize = 0.1
            stepnum = int((L-lstart)//stepsize) + 1
            stepsize = (L-lstart)/stepnum
            llist = np.linspace(lstart, L, stepnum+1)
        Eguess = (np.pi * i / llist[0]) ** 2
        E = 0
        for l in llist:
            E = (optimize.root_scalar(f=char_eq, args = (l, F), x0=Eguess, fprime=True)).root
            Eguess = E * (l/(l+stepsize))**2
        if E > E_max:
            break
        Es = np.append(Es, E)
        guess = Es[i-2] * (i/max((i-1), 1)) ** 2
        print(f"E_{i:d}={E:.2f}")
        i += 1
    return Es

def d1normFactors(Es, L, F):
    norm = Es * 0
    n = 0
    for E in Es:
        norm[n] = 1 / np.sqrt(integrate.quad(lambda x, E, F: np.abs(d1wave_fun(x, E, F)) ** 2, 0, L, args = (E, F))[0])
        print(f"N_{n+1:d}={norm[n]:.2f}")
        n += 1
    return norm

def wave_fun(x, y, basis):
    return d1wave_fun(x, d1Elevelsx[basis[1]], Fx) * d1wave_fun(y, d1Elevelsy[basis[2]], Fy) * (d1Enormsx[basis[1]] * d1Enormsy[basis[2]])

def timeEvolution(x, y, t):
    '''
    returns the wavefunction evaluated at x, y = np.meshgrid(x, y), t = t
    '''
    ret = np.zeros((y.shape[0], x.shape[0]), dtype=complex)
    i = 0
    for b in basis:
        xfun = d1wave_fun(x, d1Elevelsx[b[1]], Fx) * d1Enormsx[b[1]]
        yfun = d1wave_fun(y, d1Elevelsy[b[2]], Fy) * d1Enormsy[b[2]]
        xfun, yfun = np.meshgrid(xfun, yfun)
        ret += c0[i] * np.exp(-1j * b[0]*t) * xfun * yfun
        i += 1
    return ret

def gauss(x, y):
    return np.exp(-1/2 * ((x - x0)**2 + (y - y0)**2)/(2*d**2) + 1j*(kx*x + ky*y))

print("Energy levels:")
d1Elevelsx = d1energyLevels(max_energy, Lx, Fx)
d1basisCountx = len(d1Elevelsx)
print("Normalization factors:")
d1Enormsx = d1normFactors(d1Elevelsx, Lx, Fx)

print("Energy levels:")
d1Elevelsy = d1energyLevels(max_energy, Ly, Fy)
d1basisCounty = len(d1Elevelsy)
print("Normalization factors:")
d1Enormsy = d1normFactors(d1Elevelsy, Ly, Fy)

#basis
basis = []
for i in range(d1basisCountx):
    for j in range(d1basisCounty):
        if (d1Elevelsx[i] + d1Elevelsy[j]) < max_energy:
            basis.append((d1Elevelsx[i] + d1Elevelsy[j], i, j))

basis.sort(key=lambda a: a[0])

norm = 1 / np.sqrt(integrate.dblquad(lambda y,x: np.abs(gauss(x, y)) ** 2, 0, Lx, lambda x: 0, lambda x: Ly)[0])

'''x, y = np.meshgrid(np.linspace(0, Lx, 100), np.linspace(0, Ly, 100))
im = plt.imshow(np.abs(norm * gauss(x, y))**2, aspect='equal', origin='lower', extent=(0,Lx,0,Ly))
plt.show()'''

'''x, y = np.meshgrid(np.linspace(0, Lx, 100), np.linspace(0, Ly, 100))
for i in range(10):
    plt.imshow(np.abs(wave_fun(x, y, basis[i])) ** 2, aspect='equal', origin='lower', extent=(0,Lx,0,Ly))
    plt.colorbar()
    plt.show()'''

c0 = np.zeros(len(basis), dtype = complex)
n = 0
for b in basis:
    c0[n] = norm * (integrate.dblquad(lambda y, x: np.real(gauss(x, y) * wave_fun(x, y, b)), 0, Lx, lambda x: 0, lambda x: Ly, epsrel=1.49e-8)[0] + 1j * integrate.dblquad(lambda y, x: np.imag(gauss(x, y) * wave_fun(x, y, b)), 0, Lx, lambda x: 0, lambda x: Ly, epsrel=1.49e-8)[0])
    n += 1
    print(n , 'out of', len(basis))

print('Sum of probabilities:', np.sum(np.abs(c0) ** 2))

'''x, y = np.meshgrid(np.linspace(0, Lx, 100), np.linspace(0, Ly, 100))
im = plt.imshow(np.abs(timeEvolution(x, y, 0))**2, aspect='equal', origin='lower', extent=(0,Lx,0,Ly))
plt.show()'''

x = np.linspace(0, Lx, 200)
y = np.linspace(0, Ly, 300)
fig = plt.figure(figsize = [16, 10], dpi = 108)
im = plt.imshow(np.abs(norm * gauss(*np.meshgrid(x, y)))**2, aspect='equal', origin='lower', extent=(0,Lx,0,Ly))

def animate(i):
    data = np.abs(timeEvolution(x, y, i/FPS * timeSpeed)) ** 2
    im.set_array(data)
    print(i, 'frames out of', FPS * animTime)
    return im,

anim = animation.FuncAnimation(fig, animate, frames=animTime*FPS, interval=1000 / FPS, blit=True)

if save:
    anim.save(name, fps=FPS)
else:
    plt.show()
