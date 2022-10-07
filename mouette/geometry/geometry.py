from .vector import Vec
import numpy as np
import math
from typing import Union

def sign0(x : float):
    """sign of x
        1 if x >= 0
        -1 if x < 0

    Parameters:
        x (float): input number

    Returns:
        [float]: the sign of x
    """
    if x>=0: return 1
    return -1

def sign(x : float):
    """sign of x
        1 if x > 0
        -1 if x < 0
        0 if x = 0

    Parameters:
        x (float): input number

    Returns:
        [float]: x/|x|
    """
    if x>0: return 1
    elif x<0 : return -1
    return 0

def norm(x : np.ndarray):
    return np.sqrt(np.dot(x.flatten(), x.flatten()))

def dot(A,B):
    return np.dot(A,B)

def distance(a : np.ndarray, b : np.ndarray):
    return norm(b-a)

def cross(v1,v2):
    return Vec(
        v1[1]*v2[2] - v1[2] * v2[1],
        v2[0]*v1[2] - v2[2] * v1[0],
        v1[0]*v2[1] - v1[1] * v2[0]
    )

def cotan(A:Vec, B:Vec, C:Vec):
    """ 
    A,B,C three points in R^3
    returns the cotangent of the angle from BA to BC
    """
    A, B, C = Vec(A), Vec(B), Vec(C)
    BA = Vec.normalized(A-B)
    BC = Vec.normalized(C-B)
    cosine = np.dot(BA,BC)
    sine = norm(cross(BA,BC))
    return cosine/sine

def angle_3pts(A:Vec, B:Vec, C:Vec) -> float:
    """Angle ABC between three points

    Parameters:
        A (Vec): first point
        B (Vec): central point
        C (Vec): second point

    Returns:
        float: the angle
    """
    BA = Vec(A)-Vec(B)
    BC = Vec(C)-Vec(B)
    s = cross(BA,BC).norm()
    c = dot(BA,BC)
    return math.atan2(s, c)


def signed_angle_2vec3D(V1:Vec, V2:Vec, N:Vec) -> float:
    """Signed angle between two vectors with orientation given by normal N

    Parameters:
        V1 (Vec): First vector
        V2 (Vec): Second vector
        N (Vec): reference normal direction

    Returns:
        float: the angle
    """
    S = cross(V1,V2)
    s = S.norm()
    c = dot(V1,V2)
    return sign0(dot(S,N)) * math.atan2(s, c)

def signed_angle_3pts(A:Vec, B:Vec, C:Vec, N:Vec) -> float:
    """Signed angle between three points ABC with orientation givne by normal N

    Parameters:
        A (Vec): first point
        B (Vec): central point
        C (Vec): second point
        N (Vec): reference normal direction

    Returns:
        float: the angle
    """
    return signed_angle_2vec3D(A-B, C-B, N)

def angle_2vec2D(V1:Vec, V2:Vec) -> float:
    return math.atan2(V2.y, V2.x) - math.atan2(V1.y, V1.x)

def angle_2vec3D(V1:Vec, V2:Vec) -> float:
    C = cross(V1,V2)
    return math.atan2(C.norm(), dot(V1,V2))

def face_basis(*f):
    """ 
    Orthonormal basis of face
    Given three points A,B,C, returns a basis such that the first vector is along direction AB and third vector is normal to the plane ABC
    """
    if len(f)==1: f = f[0]
    pA,pB,pC = (x for x in f)
    X = Vec.normalized(pB-pA)
    Z = Vec.normalized(cross(X, pC-pA))
    Y = Vec.normalized(cross(Z,X))
    return X,Y,Z

def triangle_area(A:Vec, B:Vec, C:Vec) -> float :
    return cross(B-A,C-A).norm()/2

def triangle_area_2D(A:Vec, B:Vec, C:Vec) -> float :
    return abs(det_2x2(B-A,C-A)/2)

def quad_area(A:Vec, B:Vec, C:Vec, D:Vec) -> float:
    return (triangle_area(A,B,C) + triangle_area(A,C,D) + triangle_area(B,C,D) + triangle_area(B,D,A))/2

def det_2x2(A:Union[complex,np.ndarray], B:Union[complex,np.ndarray]) -> float:
    """Computes a 2x2 determinant

    Parameters:
        A (Union[complex,np.ndarray]): first column vector. Can also be a complex number
        B (Union[complex,np.ndarray]): second column vector. Can also be a complex number

    Returns:
        float: the determinant
    """
    if isinstance(A,complex):
        ax,ay = A.real, A.imag
    else:
        ax,ay = A[0], A[1]
    if isinstance(B,complex):
        bx,by = B.real, B.imag
    else:
        bx,by = B[0], B[1]
    return ax*by - ay*bx

def det_3x3(*args) -> float:
    """Computes a 3x3 using the rule of Sarrus
    
    Parameters:
        Either a 3*3 numpy array representing a matrix, or 3 3*1 numpy array representing three column vectors

    Returns:
        float: the determinant
    """
    if len(args)==1:
        mat = args[0]
        assert mat.shape == (3,3)
    elif len(args)==3:
        A,B,C = args[0], args[1], args[2]
        mat = np.array([A,B,C])
    d = mat[0,0] * mat[1,1] * mat[2,2] + \
        mat[0,1] * mat[1,2] * mat[2,0] + \
        mat[0,2] * mat[1,0] * mat[2,1] - \
        mat[0,0] * mat[1,2] * mat[2,1] - \
        mat[0,1] * mat[1,0] * mat[2,2] - \
        mat[0,2] * mat[1,1] * mat[2,0]
    return d

def intersect_2lines2D(p1 : Vec, d1 : Vec, p2: Vec, d2 : Vec) -> Vec:
    p1,d1,p2,d2 = (u[:2] for u in (p1,d1,p2,d2))
    if abs(det_2x2(d1,d2))<1e-12 : return None #parallel lines
    n2 = Vec(d2.y, -d2.x)
    t = dot(p2-p1,n2)/dot(d1,n2)
    return p1+t*d1

def circumcenter(v1 : Vec, v2 : Vec, v3: Vec) -> Vec:
    """Circumcenter of the triangle formed by three points in space

    /!\ circumcenter of triangle (v1,v2,v3) may not lay inside the triangle

    Parameters:
        v1 (Vec): first point
        v2 (Vec): second point
        v3 (Vec): third point

    Returns:
        Vec: coordinates of the circumcenter
    """
    p1 = (v1+v2)/2
    p2 = (v1+v3)/2
    d1 = v2-v1
    d2 = v3-v1
    d1 = Vec(d1.y, -d1.x)
    d2 = Vec(d2.y, -d2.x)
    S = intersect_2lines2D(p1, d1, p2, d2)
    return Vec(S.x, S.y, 0.)

def aspect_ratio(A : Vec, B : Vec, C : Vec) -> float:
    """
    Computes the aspect ratio of triangle ABC, defined as the ratio between the circumradius to twice the inradius.
    This ratio equals 1 for equilateral triangle and goes to zero as the triangle gets close to degenerate.

    Args:
        A (Vec): first point of the triangle
        B (Vec): second point of the triangle
        C (Vec): third point of the triangle

    Returns:
        float: the aspect ratio of triangle ABC

    Note:
        https://stackoverflow.com/a/10290011
    """
    ab = distance(A,B)
    bc = distance(B,C)
    ca = distance(C,A)
    s = (ab+bc+ca)/2
    return ab*bc*ca/(8*(s-ab)*(s-bc)*(s-ca))