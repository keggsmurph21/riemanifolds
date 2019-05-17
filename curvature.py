'''
Kevin Murphy
Spring 2019

Basic code for computing curvature tensors on some Riemannian manifolds, including:
 - S^n
 - R^n
 - H^n
and products of these.

Example usage:

    $ python -i curvature.py
    >>> m = Product([ S(3), H(4) ])
    >>> pretty_print(m)
    >>> m.Ric[1,1]

'''

import numpy as np
import sympy
from string import ascii_letters


def get_symbols(n):

    if n > 52:
        raise ValueError('cannot have more than 52 dimensions')

    letters = [ c for c in ascii_letters ]
    letters = letters[:n]
    letters = ' '.join(letters)

    return sympy.symbols(letters)

def pretty_print(m):

    print('\nmetric')
    print(m.G)
    print('\ninverse metric')
    print(m.G_inv)
    print('\nchristoffel symbols')
    print(m.gamma)
    print('\ndifferential christoffel symbols')
    print(m.del_gamma)
    print('\ntotal curvature tensor')
    print(m.R)
    print('\nsectional curvature')
    print(m.K)
    print('\nricci curvature')
    print(m.Ric)
    print('\nscalar curvature')
    print(m.S)


class RiemannianManifold:

    def __init__(self, dim):

        if dim < 2:
            raise ValueError('manifolds must have dimension at least 2')

        self.dim = dim
        self.x = get_symbols(dim)

        self.G = self.get_metric()
        self.G_inv = self.get_metric_inv()
        (self.gamma, self.del_gamma) = get_christoffel_symbols()
        self.R = self.get_curvature()
        self.K = self.get_sectional()
        self.Ric = self.get_ricci()
        self.S = self.get_scalar()

    def get_metric(self):

        G = np.zeros([ self.dim, self.dim ], dtype=sympy.Symbol)
        for i in range(self.dim):
            for j in range(self.dim):
                G[i,j] = self.g(i,j)

        return G

    def get_metric_inv(self):

        #G_inv = np.linalg.inv(self.G) # can't invert functions lol
        G_inv = np.zeros([ self.dim, self.dim ], dtype=sympy.Symbol)
        for i in range(self.dim):
            for j in range(self.dim):
                G_inv[i,j] = self.g_inv(i,j)

        return G_inv

    def get_christoffel_symbols(self):

        gamma = np.zeros([ self.dim, self.dim, self.dim ], dtype=sympy.Symbol)
        del_gamma = np.zeros([ self.dim, self.dim, self.dim, self.dim ], dtype=sympy.Symbol)
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):

                    expr = 0

                    for s in range(self.dim):

                        subexpr = 0

                        subexpr += sympy.diff( self.G[j,s], self.x[i] )
                        subexpr += sympy.diff( self.G[s,i], self.x[j] )
                        subexpr -= sympy.diff( self.G[i,j], self.x[s] )

                        expr += subexpr * self.G_inv[k,s]

                    expr /= 2
                    expr = sympy.simplify( expr )

                    gamma[i,j,k] = expr

                    for s in range(self.dim):
                        del_expr = sympy.diff( expr, self.x[s] )
                        del_gamma[i,j,k,s] = sympy.simplify( del_expr )

        return (gamma, del_gamma)

    def get_curvature():

        R = np.zeros([ self.dim, self.dim, self.dim, self.dim ], dtype=sympy.Symbol)
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):

                        expr = 0

                        for s in range(self.dim):

                            expr += self.gamma[i,k,s] * self.gamma[j,s,l]
                            expr -= self.gamma[j,k,s] * self.gamma[i,s,l]

                        expr += self.del_gamma[i,k,l,j]
                        expr -= self.del_gamma[j,k,l,i]
                        #expr += sympy.diff( self.gamma[i,k,l], self.x[j] )
                        #expr -= sympy.diff( self.gamma[j,k,l], self.x[i] )

                        R[i,j,k,l] = sympy.simplify( expr )

        return R

    def get_sectional():

        K = np.zeros([ self.dim, self.dim ], dtype=sympy.Symbol)
        for i in range(self.dim):
            for j in range(self.dim):

                K[i,j] = self.R[i,j,j,i]

        return K

    def get_ricci(self):

        ric = np.zeros([ self.dim, self.dim ], dtype=sympy.Symbol)
        for i in range(self.dim):
            for k in range(self.dim):
                
                expr = 0

                for j in range(self.dim):
                    expr += self.R[i,j,k,j]

                expr /= (self.dim - 1)

                ric[i,k] = sympy.simplfy( expr )

        return ric

    def get_scalar(self):

        scalar = 0
        for i in range(self.dim):
            for k in range(self.dim):
                
                scalar += self.Ric[i,k] * self.G_inv[i,k]

        scalar /= self.dim
        scalar = sympy.simplify( scalar )

        return scalar
        
    def g(self, i, j):
        raise NotImplemented('no metric specified')

    def g_inv(self, i, j):
        raise NotImplemented('no inverse metric specified')


class H(RiemannianManifold):
    
    def g(self, i, j):
       
        if i == j:
            return self.x[-1] ** (-2)
       
        return 0

    def g_inv(self, i, j):
        
        if i == j:
            return self.x[-1] ** 2
        
        return 0
    

class R(RiemannianManifold):

    def g(self, i, j):
        
        if i == j:
            return 1

        return 0

    def g_inv(self, i, j):
        return self.g(i,j)


class S(RiemannianManifold):

    def g(self, i, j):

        if i != j:
            return 0
        
        expr = 1

        for s in range(i):
            expr *= ( sympy.sin(self.x[s]) ) ** 2

        return expr

    def g_inv(self, i, j):

        if i != j:
            return 0

        return self.g(i,j) ** (-1)


class Product(RiemannianManifold):

    def __init__(self, components):

        self.x = None
        self.components = components
        self.dim = dim = sum([ M.dim for M in components])

        self.G = np.zeros([ dim, dim ], dtype=sympy.Symbol)
        self.G_inv = np.zeros([ dim, dim ], dtype=sympy.Symbol)
        self.gamma = np.zeros([ dim, dim, dim ], dtype=sympy.Symbol)
        self.del_gamma = np.zeros([ dim, dim, dim, dim ], dtype=sympy.Symbol)
        self.R = np.zeros([ dim, dim, dim, dim ], dtype=sympy.Symbol)
        self.K = np.zeros([ dim, dim ], dtype=sympy.Symbol)

        m = 0
        for M in components:

            for i in range(M.dim):
                for j in range(M.dim):

                    self.G[i+m,j+m] = M.G[i,j]
                    self.G_inv[i+m,j+m] = M.G[i,j]
                    self.K[i+m,j+m] = M.K[i,j]

                    for k in range(M.dim):

                        self.gamma[i+m,j+m,k+m] = M.gamma[i,j,k]

                        for l in range(M.dim):

                            self.del_gamma[i+m,j+m,k+m,l+m] = M.del_gamma[i,j,k,l]
                            self.R[i+m,j+m,k+m,l+m] = M.R[i,j,k,l]

            m += M.dim

        self.Ric = self.get_ricci()
        self.S = self.get_scalar()


class ImmersedManifold(RiemannianManifold):

    def __init__(self, x, fx):

        self.x = sympy.symbols(x)
        self.dim = len(self.x)

        self.sigma = fx
        self.del_sigma = np.zeros([ len(self.sigma), self.dim ], dtype=sympy.Symbol)

        for (i, coord) in enumerate(self.sigma):
            for (j, var) in enumerate(self.x):

                self.del_sigma[i,j] = sympy.diff( coord, var )

        self.G = self.get_metric()
        print(self.del_sigma)
        print(self.G)

    def g(self, i, j):

        if i == j:

            expr = 0
            for coord in self.del_sigma[:, i]:
                expr += coord * coord

            return expr

        return 0

    def g_inv(self, i, j):

        if i == j:
            return self.g(i,j) ** (-1)

        return 0

    def get_gaussian_curvature(self):
        
        expr = 1
        for i in range(len(self.x)):
            expr *= self.G[i,i]

        return sympy.simplify( expr )

    def get_mean_curvature(self):

        expr = 1
        for i in range(len(self.x)):
            expr += self.G[i,i]

        return sympy.simplify( (1/2) * expr )


if __name__ == '__main__':

    M = ImmersedManifold('x y', ('x', 'y', 'x**2 * y**2'))

    sinh = sympy.sinh
    cosh = sympy.cosh
    sin = sympy.sin
    cos = sympy.cos

    print()
    N = ImmersedManifold('u v', (
        '2*sinh(u)*cos(v) - (2/3)*sinh(3*u)*cos(3*v)',
        '2*sinh(u)*sin(v) + (2/3)*sinh(3*u)*sin(3*v)',
        '2*cosh(2*u)*cos(2*v)') )

    print()
    print(N.get_mean_curvature())

    print('valid manifolds: R(n), S(n), H(n), Product([ m1 , ... , mk ])')

