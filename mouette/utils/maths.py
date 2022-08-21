import cmath
from math import pi

def roots(c : complex, pow: int, normalize=True):
    """Given a complex number c, compute and returns the four ^(1/4) roots 
    Args:
        c (complex): input complex number
        pow (int): the power of the root. Returns power-th roots of c
        normalize (bool, optional): If True, roots will have module 1. Defaults to True

    Returns:
        list of complex roots
    """
    r,t = cmath.polar(c)
    r = 1 if normalize else r**(1/pow)
    return [cmath.rect(r, (t + 2*k*pi)/pow) for k in range(pow)]

def angle_diff(a,b):
    return (a - b + pi) % (2*pi) - pi