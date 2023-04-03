import mouette as M
from math import sin,cos, pi
from random import random

def test_principal_angle():
    for _ in range(10):
        angle_rand = 100*random()
        principal_angle = M.utils.maths.principal_angle(angle_rand)
        assert -pi<=principal_angle<=pi
        assert abs(cos(angle_rand) - cos(principal_angle))<1e-6
        assert abs(sin(angle_rand) - sin(principal_angle))<1e-6

def test_solve_quadratic():

    # A = 0
    assert M.utils.maths.solve_quadratic(0.,1.,3.) == [-3.]

    # no roots
    assert len(M.utils.maths.solve_quadratic(1.,1.,1.))==0

    # single root
    for _ in range(3):
        root = 100*random()
        B = -2*root
        C = root*root
        res = M.utils.maths.solve_quadratic(1.,B,C)
        assert len(res)==1
        assert abs(res[0]-root)<1e-6

    # two roots
    for _ in range(3):
        root1 = 100*random()
        root2 = 100*random()
        root1,root2 = min(root1,root2), max(root1,root2)
        B = -root1-root2
        C = root1*root2
        res = M.utils.maths.solve_quadratic(1.,B,C)
        assert len(res)==2
        res1,res2 = min(res), max(res)
        assert abs(res1-root1)<1e-6
        assert abs(res2-root2)<1e-6