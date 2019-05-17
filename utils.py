import sympy
import numpy as np

def cross_product(v,w):

    n = np.zeros(3, dtype=sympy.Symbol)

    n[0] = sympy.simplify( v[1] * w[2] - v[2] * w[1] )
    n[1] = sympy.simplify( v[2] * w[0] - v[0] * w[2] )
    n[2] = sympy.simplify( v[0] * w[1] - v[1] * w[0] )

    return n

def get_norm(v):

    expr = np.dot(v,v)
    expr = sympy.sqrt(expr)

    return sympy.simplify( expr )

def normalize(vec):

    norm = get_norm(vec)
    normalized = np.zeros(len(vec), dtype=sympy.Symbol)

    for (i, v) in enumerate(vec):
        normalized[i] = sympy.simplify( v / norm )

    return normalized

def det(M):

    if M.shape != (2,2):
        raise ValueError('this det() is only defined on 2x2 matrices')

    return sympy.simplify( M[0,0]*M[1,1] - M[0,1]*M[1,0] )

def trace(M):

    if M.shape != (2,2):
        raise ValueError('this trace() is only defined on 2x2 matrices')

    return sympy.simplify( M[0,0] + M[1,1] )

def invert(M):

    if M.shape != (2,2):
        raise ValueError('this invert() is only defined on 2x2 matrices')

    N = np.zeros([2,2], dtype=sympy.Symbol)
    d = 1 / det(M)

    N[0,0] = d * M[1,1]
    N[0,1] = -1 * d * M[0,1]
    N[1,0] = -1 * d * M[1,0]
    N[1,1] = d * M[0,0]

    return N

def matmul(M, N):

    if len(M.shape) > 2 or len(N.shape) > 2:
        raise ValueError('matmul() is only defined for 2-dimensional matrices')

    if M.shape[1] != N.shape[0]:
        raise ValueError(f'dimension mismatch, {M.shape} x {N.shape}')

    shape = [ M.shape[0], N.shape[1] ]
    P = np.zeros(shape, dtype=sympy.Symbol)

    for i in range(shape[0]):
        for j in range(shape[1]):

            P[i,j] = np.dot( M[i,:], N[:,j] )

    return P
