import numpy as np
from scipy import special, optimize, integrate
import pickle
from os import path

from plot import * 

class d1schroedinger:
    def __init__(self, psi0 = None, F = 1.0, L = 15.0, numPoints = 200, name = "1D: "):
        self.__name = name
        self.__F = F
        self.__L = L
        self.__numPoints = numPoints
        self.__x = np.linspace(0, L, numPoints)
        
        self.__Es = np.zeros((0))
        self.__norms = np.zeros((0))
        self.__cachedBasisFun = np.zeros((0, self.__numPoints), dtype=complex)
        self.__c0s = np.zeros((0))
        
        if psi0 != None:
            self.__unormpsi0 = psi0
            self.__psi0norm = 1 / np.sqrt(np.abs(self.scalarProd(psi0, psi0)))
            n = 0
            while True:
                self.eLevel(n)
                self.waveFunNorm(n)
                self.cacheBasisFun(n)
                self.basisCoeff(n)
                
                eWaveFunSum = np.sum(np.abs(self.__c0s)**2)
                print(self.__name + "Sum of probabilities: " + str(eWaveFunSum))
                if eWaveFunSum > 0.9999:
                    break
                n += 1
    
    @property
    def F(self):
        return self.__F
    
    @property
    def L(self):
        return self.__L
    
    @property
    def x(self):
        return self.__x
    
    @x.setter
    def x(self, a):
        self.__x = a
        n = len(self.__cachedBasisFun) - 1
        self.__cachedBasisFun = np.zeros((0, self.__numPoints), dtype=complex)
        self.cacheBasisFun(n)
        return a
    
    @property
    def Es(self):
        return np.copy(self.__Es)
    
    @property
    def norms(self):
        return np.copy(self.__norms)
    
    @property
    def c0s(self):
        return np.copy(self.__c0s)
    
    def G0(self, x, y, E):
        k = np.sqrt(E)
        prefactor = -1 / k / np.sin(k * self.__L)
        G1 = np.sin(k*x) * np.sin(k*(y-self.__L)) * (x < y)
        G2 = np.sin(k*y) * np.sin(k*(x-self.__L)) * (1 - (x < y))
        return prefactor * (G1 + G2)
    
    def G(self, x, y, E):
        # F3sqrt = np.power(self.__F, 1/3)
        # ai1, ai1p, bi1, bi1p = special.airy(-E / F3sqrt**2)
        # ai2, ai2p, bi2, bi2p = special.airy((self.__F * self.__L - E) / F3sqrt**2)
        # ai3, ai3p, bi3, bi3p = special.airy(x * F3sqrt - E / F3sqrt**2)
        # ai4, ai4p, bi4, bi4p = special.airy(y * F3sqrt - E / F3sqrt**2)
        # c2c1 = -ai1 / bi1
        # c4c3 = -ai2 / bi2
        # c3c1 = (ai4 / bi4 + c2c1) / (ai4 / bi4 + c4c3)
        # c1 = 1 / F3sqrt / ((c3c1 - 1) * ai4p + (c4c3 * c3c1 - c2c1) * bi4p)
        # c2 = c2c1 * c1
        # c3 = c3c1 * c1
        # c4 = c4c3 * c3
        # G1 = (c1 * ai3 + c2 * bi3) * (x < y)
        # G2 = (c3 * ai3 + c4 * bi3) * (1 - (x < y))
        # return G1 + G2
        F3sqrt = np.power(self.__F, 1/3)
        ai1, ai1p, bi1, bi1p = special.airy(-E / F3sqrt**2)
        ai2, ai2p, bi2, bi2p = special.airy((self.__F * self.__L - E) / F3sqrt**2)
        ai3, ai3p, bi3, bi3p = special.airy(x * F3sqrt - E / F3sqrt**2)
        ai4, ai4p, bi4, bi4p = special.airy(y * F3sqrt - E / F3sqrt**2)
        c0 = 1 / F3sqrt * np.pi / (ai1/bi1 - ai2/bi2)
        G1 = c0 * (ai4 - ai2/bi2 * bi4) * (ai3 - ai1/bi1 * bi3) * (x < y)
        G2 = c0 * (ai4 - ai1/bi1 * bi4) * (ai3 - ai2/bi2 * bi3) * (1 - (x < y))
        return G1 + G2
    
    def Galt(self, x, y, E):
        F3sqrt = np.power(self.__F, 1/3)
        ai1, ai1p, bi1, bi1p = special.airy(-E / F3sqrt**2)
        ai2, ai2p, bi2, bi2p = special.airy((self.__F * self.__L - E) / F3sqrt**2)
        ai3, ai3p, bi3, bi3p = special.airy(x * F3sqrt - E / F3sqrt**2)
        ai4, ai4p, bi4, bi4p = special.airy(y * F3sqrt - E / F3sqrt**2)
        c0 = 1 / F3sqrt * np.pi / (ai1/bi1 - ai2/bi2)
        G1 = c0 * (ai4 - ai2/bi2 * bi4) * (ai3 - ai1/bi1 * bi3) * (x < y)
        G2 = c0 * (ai4 - ai1/bi1 * bi4) * (ai3 - ai2/bi2 * bi3) * (1 - (x < y))
        return G1 + G2
    
    def saveTofile(self, filename):
        with open(filename, "wb") as outFile:
            data = [self.__name, self.__F, self.__L, self.__numPoints, self.__x, self.__Es, self.__norms, self.__cachedBasisFun, self.__c0s]
            pickle.dump(data, outFile, -1)
    
    def loadFromFile(self, filename):
        with open(filename, "rb") as inFile:
            self.__name, self.__F, self.__L, self.__numPoints, self.__x, self.__Es, self.__norms, self.__cachedBasisFun, self.__c0s = pickle.load(inFile)
    
    def cacheBasisFun(self, n):
        if len(self.__cachedBasisFun) <= n:
            for i in range(len(self.__cachedBasisFun), n+1):
                y = self.waveFun(self.__x, i)
                print(self.__name + f"{i:d}th basis function cached")
                self.__cachedBasisFun = np.append(self.__cachedBasisFun, np.array(y, copy = False, ndmin = 2), axis = 0)
                
    
    def basisCoeff(self, n):
        if len(self.__c0s) <= n:
            for i in range(len(self.__c0s), n+1):
                c0 = self.scalarProd(lambda x: self.waveFun(x, i), self.psi0)
                self.__c0s = np.append(self.__c0s, c0)
                print(self.__name + f"c_{i:d}={c0:f}")
    
    def getLength(self):
        return self.__L
    
    def scalarProd(self, a, b):
        real = integrate.quad(lambda x: np.real(np.conjugate(a(x)) * b(x)), 0, self.__L)[0]
        imag = integrate.quad(lambda x: np.imag(np.conjugate(a(x)) * b(x)), 0, self.__L)[0]
        return real + 1j * imag
    
    def psi0(self, x):
        return self.__psi0norm * self.__unormpsi0(x)
    
    def charEq(self, E, L = None):
        if L == None:
            L = self.__L
        F3sqrt = np.power(self.__F, 1/3)
        ai1, ai1p, bi1, bi1p = special.airy(-E / F3sqrt ** 2)
        ai1p /= F3sqrt ** 2
        bi1p /= F3sqrt ** 2
        ai2, ai2p, bi2, bi2p = special.airy(F3sqrt * L - E / F3sqrt ** 2)
        ai2p /= F3sqrt ** 2
        bi2p /= F3sqrt ** 2
        f = bi1*ai2 - ai1*bi2
        fp = -(bi1p*ai2 + bi1*ai2p - (ai1p*bi2 + ai1*bi2p))
        return f, fp
    
    def unormWaveFun(self, x, n):
        '''
        n goes from 0
        '''
        self.eLevel(n)
        E = self.__Es[n]
        F3sqrt = np.power(self.__F, 1/3)
        ai1, ai1p, bi1, bi1p = special.airy(-E / F3sqrt ** 2)
        ai2, ai2p, bi2, bi2p = special.airy(F3sqrt * x - E / F3sqrt ** 2)
        mask = np.array(E / F3sqrt ** 2 - F3sqrt * x > -10).astype(float)
        return (bi1 * ai2 - ai1 * bi2) * mask
    
    def waveFun(self, x, n):
        '''
        n goes from 0
        '''
        self.waveFunNorm(n)
        return self.__norms[n] * self.unormWaveFun(x, n)
    
    def eLevel(self, n):
        '''
        n goes from 0
        '''
        if len(self.__Es) <= n:
            for i in range(len(self.__Es), n+1):
                lstart = 1 / np.power(self.__F, 1/3)
                if self.__L <= lstart:
                    llist = np.array([self.__L])
                    stepsize = float("nan")
                else:
                    stepsize = 0.1
                    stepnum = int((self.__L-lstart)//stepsize) + 1
                    stepsize = (self.__L-lstart)/stepnum
                    llist = np.linspace(lstart, self.__L, stepnum+1)
                Eguess = (np.pi * (i+1) / llist[0]) ** 2
                E = 0
                for l in llist:
                    E = (optimize.root_scalar(f=self.charEq, args = (l), x0=Eguess, fprime=True)).root
                    Eguess = E * (l/(l+stepsize))**2
                print(self.__name + f"E_{i:d}={E:.2f}")
                self.__Es = np.append(self.__Es, E)
        return
    
    def waveFunNorm(self, n):
        '''
        n goes from 0
        '''
        F3sqrt = np.power(self.F, 1/3)
        if len(self.__norms) <= n:
            for i in range(len(self.__norms), n+1):
                self.eLevel(i)
                ai1, ai1p, bi1, bi1p = special.airy(-self.Es[i] / F3sqrt**2)
                ai2, ai2p, bi2, bi2p = special.airy(self.L * F3sqrt - self.Es[i] / F3sqrt**2)
                intsquared = 1 / F3sqrt * (1 / np.pi**2 - (bi1*ai2p - ai1*bi2p * (self.Es[i] - self.L * self.F > -10))**2)
                norm = 1 / np.sqrt(intsquared)
                print(self.__name + f"N_{i:d}={norm:.2f}")
                self.__norms = np.append(self.__norms, norm)
        return
    
    def timeEvolution(self, t = 0, x = None):
        if x != None:
            ret = np.zeros(x.shape, dtype = complex)
            for n in range(len(self.__Es)):
                ret += self.__c0s[n] * np.exp(-1j * self.__Es[n]*t) * self.waveFun(x, n)
        else:
            ret = np.zeros((self.__numPoints), dtype = complex)
            for n in range(len(self.__cachedBasisFun)):
                ret += self.__c0s[n] * np.exp(-1j * self.__Es[n]*t) * self.__cachedBasisFun[n, :]
            
        return ret

class d2schroedinger:
    def __init__(self, psi0 = None, Fx = 0.00001, Fy = 0.00001, Lx = 10, Ly = 15, numPoints = 200, name = "2D  : "):
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        x, y = np.meshgrid(x, y)
        if psi0 != None:
            res = x * y**2 * psi0(x, y)
        else:
            res = np.zeros((0))
        filename =  (str(np.sum(np.abs(res))) + str(Fx) + str(Fy) + str(Lx) + str(Ly) + str(numPoints) + name + ".2dcache").replace(" ", "")
        directory = "../cache/"
        if path.isfile(directory + filename):
            with open(directory + filename, "rb") as infile:
                self.loadFromFile(directory + filename)
        else:
            self.__name = name
            self.__Fx = Fx
            self.__Fy = Fy
            self.__Lx = Lx
            self.__Ly = Ly
            
            self.__d1x = d1schroedinger(F=Fx, L=Lx, numPoints=numPoints, name = "1D x: ")
            self.__d1y = d1schroedinger(F=Fy, L=Ly, numPoints=numPoints, name = "1D y: ")
            
            self.__Es = np.zeros((0))
            self.__qNums = np.zeros((0, 2), dtype=int)
            self.__cachedBasisFun = np.zeros((0, self.__d1y.x.shape[0], self.__d1x.x.shape[0]), dtype=complex)
            self.__c0s = np.zeros((0), dtype=complex)
        
        if psi0 != None:
            self.__unormPsi0 = psi0
            self.__psi0norm = 1 / np.sqrt(self.scalarProd(psi0, psi0))
            
            n = 0
            while True:
                self.eLevel(n)
                self.cacheBasisFun(n)
                self.basisCoeff(n)
                
                eWaveFunSum = np.sum(np.abs(self.__c0s)**2)
                print(self.__name + f"Sum of probabilities: {eWaveFunSum:f}")
                if eWaveFunSum > 0.99:
                    break
                n += 1
        self.saveToFile(directory + filename)
        
    @property
    def Es(self):
        return np.copy(self.__Es)
    
    @property
    def x(self):
        return self.__d1x.x
    
    @x.setter
    def x(self, a):
        self.__d1x.x = a
        n = len(self.__cachedBasisFun) - 1
        self.__cachedBasisFun = np.zeros((0, self.__d1y.x.shape[0], self.__d1x.x.shape[0]), dtype=complex) 
        self.cacheBasisFun(n)
        return a
        
    @property
    def y(self):
        return self.__d1y.x
    
    @y.setter
    def y(self, a):
        self.__d1y.x = a
        n = len(self.__cachedBasisFun) - 1
        self.__cachedBasisFun = np.zeros((0, self.__d1y.x.shape[0], self.__d1x.x.shape[0]), dtype=complex) 
        self.cacheBasisFun(n)
        return a
    
    @property
    def Lx(self):
        return self.__Lx
    
    @property
    def Ly(self):
        return self.__Ly
    
    @property
    def Fx(self):
        return self.__Fx
    
    @property
    def Fy(self):
        return self.__Fy
    
    @property
    def cachedBasisFun(self):
        return np.copy(self.__cachedBasisFun)
    
    def saveToFile(self, filename):
        with open(filename, "wb") as outFile:
            data = [self.__name, self.__Fx, self.__Fy, self.__Lx, self.__Ly, self.__Es, self.__qNums, self.__cachedBasisFun, self.__c0s]
            pickle.dump(data, outFile, -1)
            self.__d1x.saveTofile(filename + ".d1x")
            self.__d1y.saveTofile(filename + ".d1y")
    
    def loadFromFile(self, filename):
        with open(filename, "rb") as inFile:
            self.__name, self.__Fx, self.__Fy, self.__Lx, self.__Ly, self.__Es, self.__qNums, self.__cachedBasisFun, self.__c0s = pickle.load(inFile)
            self.__d1x = d1schroedinger()
            self.__d1x.loadFromFile(filename + ".d1x")
            self.__d1y = d1schroedinger()
            self.__d1y.loadFromFile(filename + ".d1y")
    
    def scalarProd(self, a, b):
        real = integrate.dblquad(lambda y, x: np.real(np.conjugate(a(x, y)) * b(x, y)), 0, self.__Lx, lambda x: 0, lambda x: self.__Ly)[0]
        imag = integrate.dblquad(lambda y, x: np.imag(np.conjugate(a(x, y)) * b(x, y)), 0, self.__Lx, lambda x: 0, lambda x: self.__Ly)[0]
        return real + 1j * imag
    
    def eLevel(self, n):
        if len(self.__Es) == 0:
            self.__d1x.eLevel(0)
            self.__d1y.eLevel(0)
            Ex = self.__d1x.Es[0]
            Ey = self.__d1y.Es[0]
            self.__Es = np.append(self.__Es, Ex+Ey)
            self.__qNums = np.append(self.__qNums, np.array([0, 0], dtype=int, ndmin=2), axis=0)
        if len(self.__Es) <= n:
            for i in range(len(self.__Es), n+1):
                Eprev = self.__Es[i-1]
                while self.__d1x.Es[-1] < Eprev:
                    self.__d1x.eLevel(len(self.__d1x.Es))
                while self.__d1y.Es[-1] < Eprev:
                    self.__d1y.eLevel(len(self.__d1y.Es))
                xEs = self.__d1x.Es
                yEs = self.__d1y.Es
                Emin = float("inf")
                for indx, xE in zip(range(len(xEs)), xEs):
                    for indy, yE in zip(range(len(yEs)), yEs):
                        if Eprev < xE + yE:
                            if xE + yE < Emin:
                                Emin = xE + yE
                                qNum = [indx, indy]
                            break
                self.__Es = np.append(self.__Es, Emin)
                self.__qNums = np.append(self.__qNums, np.array(qNum, dtype=int, ndmin=2), axis=0)
                print(self.__name + f"E_{i}={Emin:f}, quantum numbers {qNum}")
    
    def waveFun(self, x, y, n):
        self.eLevel(n)
        return self.__d1x.waveFun(x, self.__qNums[n, 0]) * self.__d1y.waveFun(y, self.__qNums[n, 1])
    
    def cacheBasisFun(self, n):
        if len(self.__cachedBasisFun) <= n:
            for i in range(len(self.__cachedBasisFun), n+1):
                x = self.__d1x.x
                y = self.__d1y.x
                self.eLevel(i)
                xn = self.__qNums[i, 0]
                yn = self.__qNums[i, 1]
                xfun = self.__d1x.waveFun(x, xn)
                yfun = self.__d1y.waveFun(y, yn)
                xfun, yfun = np.meshgrid(xfun, yfun)
                basisFun = xfun * yfun
                self.__cachedBasisFun = np.append(self.__cachedBasisFun, np.array(basisFun, ndmin=3), axis=0)
    
    def basisCoeff(self, n):
        if len(self.__c0s) <= n:
            for i in range(len(self.__c0s), n+1):
                c0 = self.scalarProd(lambda x, y: self.waveFun(x, y, i), self.normPsi0)
                self.__c0s = np.append(self.__c0s, c0)
                print(self.__name + f"c_{i:d}={c0:f}")
    
    def normPsi0(self, x, y):
        return self.__psi0norm * self.__unormPsi0(x, y)
    
    def timeEvolution(self, t, x = None, y = None):
        if x != None and y != None:
            ret = 0 * x
            for i in range(len(self.__Es)):
                ret += (self.__c0s[i] * np.exp(-1j * self.__Es[i] * t)) * self.waveFun(x, y, i)
        else:
            ret = 0 * self.__cachedBasisFun[0]
            for i in range(len(self.__cachedBasisFun)):
                ret += (self.__c0s[i] * np.exp(-1j * self.__Es[i] * t)) * self.__cachedBasisFun[i]
        return ret
