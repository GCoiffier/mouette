import pygeomesh as GEO
from pygeomesh import geometry as geom
from math import *
import numpy as np

def test_sign():
    assert geom.sign(1.)==1
    assert geom.sign(-1.)==-1
    assert geom.sign(2.45)==1
    assert geom.sign(-1e8)==-1
    assert geom.sign(0.)==0

def test_norm():
    x = GEO.Vec(1.,1.,1.)
    assert geom.norm(x) == x.norm()
    assert abs(geom.norm(x) - sqrt(3))<1e-15
    x = GEO.Vec(3.,4.)
    assert geom.norm(x) == 5.
    a = np.array([[1,0],[0,1]])
    assert abs(geom.norm(a) - sqrt(2)) < 1e-15


def test_dot():
    x = GEO.Vec(1.,1.,1.)
    y = GEO.Vec(0.,1.,0.)
    assert geom.dot(x,y) == 1.
    y = GEO.Vec(-1,2,1)
    assert geom.dot(x,y) == 2.

def test_distance():
    x = GEO.Vec(1.,1.,1.)
    y = GEO.Vec(1.,0.,1.)
    assert geom.distance(x,y) == 1.

def test_cross():
    x = GEO.Vec(1.,0.,0.)
    y = GEO.Vec(0.,1.,0.)
    assert (geom.cross(x,y) == GEO.Vec(0.,0.,1.)).all()

def test_cotan():
    A = GEO.Vec(1.,0.,0.)
    B = GEO.Vec(0.,0.,0.)
    C = GEO.Vec(0.,1.,0.)
    assert geom.cotan(A,B,C)==0.

def test_angle3pts():
    A = GEO.Vec.random(3)
    B = GEO.Vec.random(3)
    C = GEO.Vec.random(3)
    ang = geom.angle_3pts(A,B,C)
    assert True

def test_signed_angle_2vec3D():
    A = GEO.Vec.random(3)
    B = GEO.Vec.random(3)
    N = GEO.Vec.random(3)
    ang = geom.signed_angle_2vec3D(A,B,N)
    assert True

def test_signed_angle_3pts():
    A = GEO.Vec.random(3)
    B = GEO.Vec.random(3)
    N = GEO.Vec.random(3)
    ang = geom.signed_angle_2vec3D(A,B,N)
    assert True

def test_angle_2vec2d():
    A = GEO.Vec(1.,0.)
    B = GEO.Vec(1.,1.)
    assert geom.angle_2vec2D(A,B) == pi/4

def test_angle_2vec3d():
    A = GEO.Vec(1.,0.,0.)
    B = GEO.Vec(0.,1.,0.)
    assert geom.angle_2vec3D(A,B) == pi/2

def test_face_basis():
    A = GEO.Vec.random(3)
    B = GEO.Vec.random(3)
    C = GEO.Vec.random(3)
    X,Y,Z = geom.face_basis(A,B,C)
    X,Y,Z = geom.face_basis((A,B,C))
    assert True

def test_triangle_area():
    A = GEO.Vec.random(3)
    B = GEO.Vec.random(3)
    C = GEO.Vec.random(3)
    assert geom.triangle_area(A,B,C)>=0

    A = GEO.Vec.random(2)
    B = GEO.Vec.random(2)
    C = GEO.Vec.random(2)
    assert geom.triangle_area_2D(A,B,C)>=0

def test_quad_area():
    A = GEO.Vec(0.,0.,0.)
    B = GEO.Vec(1.,0.,0.)
    C = GEO.Vec(1.,1.,0.)
    D = GEO.Vec(0.,1.,0.)
    assert geom.quad_area(A,B,C,D)==1.

def test_det_2x2():
    A = GEO.Vec(1.,0.)
    B = GEO.Vec(0.,1.)
    assert geom.det_2x2(A,B)==1.

def test_det_3x3():
    A = GEO.Vec(1.,0.,0.)
    B = GEO.Vec(0.,1.,0.)
    C = GEO.Vec(0.,0.,1.)
    assert geom.det_3x3(A,B,C)==1.
    assert geom.det_3x3(A,C,B)==-1.

def test_intersect_2lines2D():
    x = geom.intersect_2lines2D(GEO.Vec.zeros(3), GEO.Vec(1.,0.,0.), GEO.Vec.zeros(3), GEO.Vec(1.,1.,0.))
    assert geom.norm(x)<1e-15
    dir = GEO.Vec.random(3)
    assert geom.intersect_2lines2D(GEO.Vec(1.,0.,0.), dir, GEO.Vec(-1.,0.,0.), dir) is None

def test_circumcenter():
    A = GEO.Vec(-1,0.,0.)
    B = GEO.Vec(1.,0.,0.)
    alpha = 2*pi*np.random.random()
    C = GEO.Vec(cos(alpha), sin(alpha), 0.)
    assert geom.norm(geom.circumcenter(A,B,C))<1e-10