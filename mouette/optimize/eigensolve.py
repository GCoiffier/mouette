import scipy.sparse as sp
import numpy as np
from .. import geometry as geom

def inverse_power_method(A : sp.csc_matrix, B: sp.csc_matrix = None, maxiter:int = 100, tol:float=1e-10) -> np.ndarray:
    """
    Implementation of the inverse power method (or inverse iteration) scheme to find an eigenvector associated with the smallest eigenvalue of A.
    In other words, this function computes x such that : Ax = λBx where λ is the smallest (non-zero) eigenvalue (in magnitude) of A.

    Args:
        A (sp.csc_matrix): the matrix 
        B (sp.csc_matrix, optional): metrics matrix. If not specified, will be the identity matrix. Defaults to None.
        maxiter (int, optional): maximal number of internal iteration. Defaults to 100.
        tol (float, optional): early stopping criterion. Will stop the iteration if |x_{n+1} - x_n| < tol. Defaults to 1e-6.

    Returns:
        np.ndarray: _description_
    """
    n = A.shape[0]
    B = sp.eye(n, format="csc") if B is None else B
    solve = sp.linalg.factorized(A)
    x = np.random.random(n)

    A_is_hermitian = (A.dtype==complex)
    if A_is_hermitian:
        x = np.sqrt(np.random.uniform(0, 1, n)) * np.exp(1.j * np.random.uniform(0, 2 * np.pi, n)) # values in unit disk in complex plane
    else:
        x = 2*np.random.random(n)-1 # values in [-1,1]

    it = 0
    stop_criterion = False
    while not stop_criterion:
        newx = solve(B@x)
        newxT = newx.conj() if A_is_hermitian else newx
        newx /= np.sqrt(np.dot(newxT, B@newx))
        it += 1
        stop_criterion = (it>=maxiter or geom.distance(newx, x)<tol)
        x = newx
    return x
