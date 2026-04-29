import cmath
from math import pi, sqrt

def roots(c : complex, pow: int, normalize=True):
    """Given a complex number c, compute and returns the four ^(1/4) roots 
    Parameters:
        c (complex): input complex number
        pow (int): the power of the root. Returns power-th roots of c
        normalize (bool, optional): If True, roots will have module 1. Defaults to True

    Returns:
        list of complex roots
    """
    r,t = cmath.polar(c)
    r = 1 if normalize else r**(1/pow)
    return [cmath.rect(r, (t + 2*k*pi)/pow) for k in range(pow)]

def angle_diff(a : float, b : float) -> float:
    return (a - b + pi) % (2*pi) - pi

def principal_angle(a : float) -> float :
    """
    From an arbitrary angle value, returns the equivalent angle which values lays in [-pi, pi[
    """
    b = a%(2*pi)
    if b>pi:
        b-=2*pi
    return b

def solve_quadratic(A : float, B : float, C : float):
    """
    Solves Ax² + Bx + C = 0 for real-valued roots

    Args:
        A (float): coefficient of X² 
        B (float): coefficient of X
        C (float): constant coefficient

    Returns:
        list: a list containing 0, 1 or 2 roots
    """
    if A==0: # linear case
        if B == 0:
            return []
        return [-C/B]
    delta = B*B-4*A*C
    if delta<0:
        return []
    if abs(delta)<1e-14: # delta = 0
        return [-B/(2*A)]
    else:
        delta = sqrt(delta)
        return [ (-B + delta)/(2*A), (-B - delta)/(2*A)]
