import numpy as np
from scipy import special, optimize, integrate

print("importing stuff")

class d1schroedinger:
    '''
    __F
    __L
    __psi0
    __psinorm
    
    __Es
    __norms
    __c0s
    '''
    
    def __init__(self, psi0, F = 1, L = 15, numPoints = 200):
        self.__F = F
        self.__L = L
        self.__numPoints = numPoints
        self.__unormpsi0 = psi0
        self.__psi0norm = 1 / np.sqrt(np.abs(self.scalarProd(psi0, psi0)))
        
        self.__Es = np.zeros((0))
        self.__norms = np.zeros((0))
        self.__c0s = np.zeros((0))
        self.__cachedBasisFun = np.zeros((0, self.__numPoints), dtype=complex)
        
        n = 0
        while True:
            E = self.eLevel(n)
            self.__Es = np.append(self.__Es, E)
            norm = self.waveFunNorm(n)
            self.__norms = np.append(self.__norms, norm)
            x = np.linspace(0, self.__L, self.__numPoints)
            y = self.waveFun(x, n)
            self.__cachedBasisFun = np.append(self.__cachedBasisFun, np.array(y, copy = False, ndmin = 2), axis = 0)
            c0 = self.scalarProd(lambda x: self.waveFun(x, n), self.psi0)
            self.__c0s = np.append(self.__c0s, c0)
            
            eWaveFunNorm = np.sqrt(np.sum(np.abs(self.__c0s)**2))
            print(f"Norm of energy wave function: {eWaveFunNorm:f}")
            if eWaveFunNorm > 0.9999:
                break
            n += 1
    
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
        return self.__norms[n] * self.unormWaveFun(x, n)
    
    def eLevel(self, n):
        '''
        n goes from 0
        '''
        lstart = 1 / np.power(self.__F, 1/3)
        if self.__L <= lstart:
            llist = np.array([self.__L])
            stepsize = float("nan")
        else:
            stepsize = 0.1
            stepnum = int((self.__L-lstart)//stepsize) + 1
            stepsize = (self.__L-lstart)/stepnum
            llist = np.linspace(lstart, self.__L, stepnum+1)
        Eguess = (np.pi * (n+1) / llist[0]) ** 2
        E = 0
        for l in llist:
            E = (optimize.root_scalar(f=self.charEq, args = (l), x0=Eguess, fprime=True)).root
            Eguess = E * (l/(l+stepsize))**2
        print(f"E_{n:d}={E:.2f}")
        return E
    
    def waveFunNorm(self, n):
        '''
        n goes from 0
        '''
        phin = lambda x: self.unormWaveFun(x, n)
        norm = 1 / np.sqrt(np.abs(self.scalarProd(phin, phin)))
        print(f"N_{n:d}={norm:.2f}")
        return norm
    
    def timeEvolution(self, t = 0, x = None):
        if x != None:
            ret = np.zeros(x.shape, dtype = complex)
            for n in range(len(self.__Es)):
                ret += self.__c0s[n] * np.exp(-1j * self.__Es[n]*t) * self.waveFun(x, n)
        else:
            ret = np.zeros((self.__numPoints), dtype = complex)
            for n in range(len(self.__Es)):
                ret += self.__c0s[n] * np.exp(-1j * self.__Es[n]*t) * self.__cachedBasisFun[n, :]
            
        return ret