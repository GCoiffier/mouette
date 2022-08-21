from .vector import Vec
import numpy as np
import math

def sign(x : float):
    """sign of x
        1 if x > 0
        -1 if x < 0
        0 if x = 0

    Args:
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

    Args:
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

    Args:
        V1 (Vec): First vector
        V2 (Vec): Second vector
        N (Vec): reference normal direction

    Returns:
        float: the angle
    """
    S = cross(V1,V2)
    s = S.norm()
    c = dot(V1,V2)
    return sign(dot(S,N)) * math.atan2(s, c)

def signed_angle_3pts(A:Vec, B:Vec, C:Vec, N:Vec) -> float:
    """Signed angle between three points ABC with orientation givne by normal N

    Args:
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

def det_2x2(A:np.ndarray, B:np.ndarray):
    return A[0]*B[1] - A[1]*B[0]

def det_3x3(*args):
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

    Args:
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