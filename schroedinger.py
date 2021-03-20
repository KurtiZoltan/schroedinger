import numpy as np
from scipy import special, optimize, integrate

class d1schroedinger:
    def __init__(self, psi0 = None, F = 1, L = 15, numPoints = 200, x = None):
        self.__F = F
        self.__L = L
        self.__numPoints = numPoints
        if x == None:
            self.__x = np.linspace(0, L, numPoints)
        else:
            self.__x = x
        
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
                print(f"Sum of probabilities: {eWaveFunSum:f}")
                if eWaveFunSum > 0.9999:
                    break
                n += 1
    
    @property
    def L(self):
        return self.__L
    
    def cacheBasisFun(self, n):
        if len(self.__cachedBasisFun) <= n:
            for i in range(len(self.__cachedBasisFun), n+1):
                y = self.waveFun(self.__x, i)
                print(f"{i:d}th basis function cached")
                self.__cachedBasisFun = np.append(self.__cachedBasisFun, np.array(y, copy = False, ndmin = 2), axis = 0)
                
    
    def basisCoeff(self, n):
        if len(self.__c0s) <= n:
            for i in range(len(self.__c0s), n+1):
                c0 = self.scalarProd(lambda x: self.waveFun(x, i), self.psi0)
                self.__c0s = np.append(self.__c0s, c0)
                print(f"c_{i:d}={c0:f}")
    
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
                print(f"E_{i:d}={E:.2f}")
                self.__Es = np.append(self.__Es, E)
        return
    
    def waveFunNorm(self, n):
        '''
        n goes from 0
        '''
        if len(self.__norms) <= n:
            for i in range(len(self.__norms), n+1):
                phin = lambda x: self.unormWaveFun(x, i)
                norm = 1 / np.sqrt(np.abs(self.scalarProd(phin, phin)))
                print(f"N_{i:d}={norm:.2f}")
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
    def __init__(psi0 = None, Fx = 0.00001, Fy = 0.00001, Lx = 10, Ly = 15):
        self.__Fx = Fx
        self.__Fy = Fy
        self.__Lx = Lx
        self.__Ly = Ly
        
        
        