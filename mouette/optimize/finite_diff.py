import numpy as np

def estimate_gradient_finite_differentes(fun, X : np.ndarray, eps: float = 1e-8):
    grad_FD = np.zeros_like(X)
    nvar = X.shape[0]
    for i in range(nvar):
        X[i] += eps
        ep = fun(X)
        X[i] -= 2*eps
        em = fun(X)
        X[i] += eps
        grad_FD[i] = (ep - em)/(2*eps)
    return grad_FD