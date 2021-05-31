import numpy as np
from matplotlib import animation
from schroedinger import *
from classical import *
from plot import *


# Fx = 0.0001
# Fy = 1
# Lx = 30
# Ly = 10
# x0 = 5
# y0 = 3
# d = np.sqrt(2)
# kx = 0
# ky = 3


Fx = 0.00001
Fy = 1
Lx = 10
Ly = 10
x0 = 5
y0 = 5
d = 2
kx = 0.75
ky = 1.5

animTime = 30
FPS = 30
timeSpeed = 0.5

name = 'final2d.mp4'
save = False

def gauss(x, y):
    return np.exp(-((x - x0)**2 + (y - y0)**2)/(2*d**2) + 1j*(kx*x + ky*y))

test = d2schroedinger(gauss, Lx=Lx, Ly=Ly, Fx = Fx, Fy = Fy, numPoints=300)
point = classicalPoint(x0, y0, kx, ky, test.Lx, test.Ly, test.Fx, test.Fy)

x = test.x
y = test.y
x, y = np.meshgrid(x, y)
plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title
fig = plt.figure(figsize = [16, 10], dpi = 108)
plt.xlabel("$ax$")
plt.ylabel("$ay$")
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

plt.colorbar()
if save:
    anim.save("../videos/" + name, fps=FPS)
else:
    plt.show()
