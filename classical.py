import numpy as np
epsilon = 1e-6

def check1dCollision(a, v, h, h0, collt):
    iscoll = False
    disc = v**2 - 2*a*(h-h0)
    if disc > 0:
        t1 = (-v - np.sqrt(disc)) / a
        t2 = (-v + np.sqrt(disc)) / a
    else:
        t1 = t2 = float("nan")
    if epsilon < t1 and t1 < collt:
        collt = t1
        iscoll = True
    if epsilon < t2 and t2 < collt:
        collt = t2
        iscoll = True
    return collt, iscoll

class classicalPoint:
    def __init__(self, x0, y0, kx0, ky0, Lx, Ly, Fx, Fy, m=1/2):
        self.__x = x0
        self.__y = y0 
        self.__vx = kx0 / m
        self.__vy = ky0 / m
        self.__ax = -Fx / m
        self.__ay = -Fy / m
        self.__Lx = Lx
        self.__Ly = Ly
    
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    def step(self, dt):
        while True:
            collt = dt
            collt, collx1 = check1dCollision(self.__ax, self.__vx, self.__x, 0, collt)
            collt, collx2 = check1dCollision(self.__ax, self.__vx, self.__x, self.__Lx, collt)
            collt, colly1 = check1dCollision(self.__ay, self.__vy, self.__y, 0, collt)
            collt, colly2 = check1dCollision(self.__ay, self.__vy, self.__y, self.__Ly, collt)
            self.__x += self.__vx * collt + self.__ax / 2 * collt**2
            self.__y += self.__vy * collt + self.__ay / 2 * collt**2
            self.__vx += self.__ax * collt
            self.__vy += self.__ay * collt
            if colly1 or colly2:
                self.__vy *= -1
            elif collx1 or collx2:
                self.__vx *= -1
            else:
                break
            dt -= collt
    