import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from schroedinger import *
from classical import *

Fx = 1
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

name = 'messenger.mp4'
save = False

def gauss(x, y):
    return np.exp(-1/2 * ((x - x0)**2 + (y - y0)**2)/(2*d**2) + 1j*(kx*x + ky*y))

test = d2schroedinger(gauss, Fx = Fx, Fy = Fy)
point = classicalPoint(x0, y0, kx, ky, test.Lx, test.Ly, test.Fx, test.Fy)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "New Century Schoolbook", "Bookman", "Computer Modern Roman"],
    "font.sans-serif": ["Helvetica", "Avant Garde", "Computer Modern Sans serif"],
    "font.cursive": "Zapf Chancery",
    "font.monospace": ["Courier", "Computer Modern Typewriter"],
    "text.usetex": True,
    "lines.linewidth": 0.8,
    'figure.autolayout': True})

x = test.x
y = test.y
x, y = np.meshgrid(x, y)
fig = plt.figure(figsize = [16, 10], dpi = 108)
im = plt.imshow(np.abs(test.normPsi0(x, y))**2, aspect='equal', origin='lower', extent=(0,test.Lx,0,test.Ly))
ball, = plt.plot(point.x, point.y, "ro")

def animate(i):
    time = i/FPS * timeSpeed
    data = np.abs(test.timeEvolution(time)) ** 2
    point = classicalPoint(x0, y0, kx, ky, test.Lx, test.Ly, test.Fx, test.Fy)
    point.step(time)
    im.set_array(data)
    ball.set_data(point.x, point.y)
    if save:
        print(i, 'frames out of', FPS * animTime)
    return [im, ball]

anim = animation.FuncAnimation(fig, animate, frames=animTime*FPS, interval=1000 / FPS, blit=True)

if save:
    anim.save(name, fps=FPS)
else:
    plt.show()

t = np.linspace(0, 30, 100)
x = np.zeros(t.shape)

for i, currt in zip(range(len(t)), t):
    state = lambda x, y: test.timeEvolution(currt, x, y)
    xstate = lambda x, y: x * test.timeEvolution(currt, x, y)
    temp = test.scalarProd(state, xstate)
    x[i] = np.real(temp)
    print(i, np.imag(temp))
np.savetxt("t.txt", t)
np.savetxt("x.txt", x)
plt.plot(t, x)
plt.show()