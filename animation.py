import matplotlib.pyplot as plt
import numpy as np
from scipy import special, optimize, integrate
from matplotlib import animation

F = 3
L = 15
max_energy = 30

x0 = 5
d = 1
k = 0.1

animTime = 120
FPS = 30
timeSpeed = 0.8

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

def timeEvolution(x, t):
    ret = np.zeros(x.shape, dtype = complex)
    for i in range(d1basisCount):
        ret += c0[i] * np.exp(-1j * d1Elevels[i]*t) * d1Enorms[i] * d1wave_fun(x, d1Elevels[i], F)
    return ret

def gauss(x):
    return np.exp(-1/2 * ((x - x0)/(d))**2 + 1j*k*x)
    #return x * (L - x)

print("Energy levels:")
d1Elevels = d1energyLevels(max_energy, L, F)
d1basisCount = len(d1Elevels)
print("Normalization factors:")
d1Enorms = d1normFactors(d1Elevels, L, F)


'''errors = np.zeros((basisCount, basisCount))
for i in range(basisCount):
    for j in range(i, basisCount):
        errors[i, j] = integrate.quad(lambda x: Enorms[i]*wave_fun(x, Elevels[i]) * Enorms[j]*wave_fun(x, Elevels[j]), 0, L)[0]
        if i == j:
            errors[i, j] -= 1
print("Max error in basis scalar products:", np.max(np.abs(errors)))'''

'''i = 3
j = 99
print(integrate.quad(lambda x: Enorms[i]*wave_fun(x, Elevels[i]) * Enorms[j]*wave_fun(x, Elevels[j]), 0, L)[0])'''

'''plt.figure(figsize = [5, 4])
x = np.linspace(Elevels[0]*0.95, Elevels[-1]*1.05, 1000)
y = char_eq(x, L)[0]
plt.plot(x, y)
plt.vlines(Elevels, np.min(y), np.max(y))
plt.grid()
plt.show()'''

'''plt.figure(figsize = [5, 4])
x = np.linspace(0, L, 1000)
for i in range(3):
    y = d1Enorms[i] * d1wave_fun(x, d1Elevels[i], F)
    plt.plot(x, y)
plt.grid()
plt.show()'''

norm = 1 / np.sqrt(integrate.quad(lambda x: np.abs(gauss(x)) ** 2, 0, L)[0])
c0 = np.zeros(d1basisCount, dtype = complex)
n = 0
for E in d1Elevels:
    c0[n] = norm * (integrate.quad(lambda x: np.real(gauss(x) * np.conjugate(d1Enorms[n]*d1wave_fun(x, d1Elevels[n], F))), 0, L)[0] + 1j * integrate.quad(lambda x: np.imag(gauss(x) * np.conjugate(d1Enorms[n]*d1wave_fun(x, d1Elevels[n], F))), 0, L)[0])
    n += 1

print("precision:", np.sum(np.abs(c0)**2))
#print(np.abs(c0)**2)
'''plt.plot(np.real(c0), np.imag(c0))
plt.axis("equal")
plt.show()'''

'''plt.figure(figsize = [5, 4])
x = np.linspace(0, L, 1000)
y1 = norm * gauss(x)
y2 = timeEvolution(x, 0)
plt.plot(x, np.real(y1))
plt.plot(x, np.imag(y1))
plt.grid()
plt.show()'''

'''plt.figure(figsize = [5, 4])
x = np.linspace(0, L, 1000)
y1 = norm * gauss(x)
y2 = timeEvolution(x, 0)
plt.plot(x, np.real(y1 - y2))
plt.plot(x, np.imag(y1 - y2))
plt.grid()
plt.show()'''

fig = plt.figure(figsize = [5, 4], dpi = 300)
ax = plt.axes(xlim=(0, L), ylim=(0, 1))
line, = ax.plot([], [], lw=1)
plt.grid()

def animate(i):
    x = np.linspace(0, L, 200)
    y = np.abs(timeEvolution(x, i/FPS * timeSpeed)) ** 2
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, frames=animTime*FPS, interval=1000 / FPS, blit=True)
if save:
    anim.save('pingpong.mp4', fps=FPS, extra_args=['-vcodec', 'libx264'])
else:
    plt.show()