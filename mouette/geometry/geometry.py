from .vector import Vec

import numpy as np
from numpy import dot
from numpy.linalg import norm

from numba import njit
import math
from typing import Union

@njit
def sign0(x: float):
    """
    $$\\text{sign0}(x)=\\begin{cases} 
        1 \\text{ if } x\\geqslant0 \\\\
        -1 \\text{ if } x<0
    \\end{cases}$$

    Parameters:
        x (float): input number

    Returns:
        float: the sign of x
    """
    if x>=0: return 1
    return -1

@njit
def sign(x: float):
    """
    $$\\text{sign}(x)=\\begin{cases} 
        0 \\text{ if } x=0 \\\\
        1 \\text{ if } x>0 \\\\
        -1 \\text{ if } x<0
    \\end{cases}$$

    Parameters:
        x (float): input number

    Returns:
        float: x/|x| and 0 for x=0
    """
    if x>0: return 1
    elif x<0 : return -1
    return 0


@njit
def distance2(A: Vec, B: Vec) -> float:
    """Squared euclidean distance between points A and B.

    Args:
        A (Vec): first vector
        B (Vec): second point

    Returns:
        float : distance squared.
    """
    return dot(B-A, B-A)

@njit
def distance(A: Vec, B: Vec) -> float:
    """Distance between A and B.

    Args:
        A (Vec): first point
        B (Vec): second point
        
    Returns:
        float: distance
    """
    return norm(B-A)

@njit
def cross(A: Vec, B: Vec) -> Vec:
    """Cross product of vectors A and B

    Args:
        A (Vec): Size 3 vector
        B (Vec): Size 3 vector
        
    Returns:
        Vec: cross product AxB
    """
    return np.array((
        A[1]*B[2] - A[2] * B[1],
        B[0]*A[2] - B[2] * A[0],
        A[0]*B[1] - A[1] * B[0]
    ))

@njit
def cotan(A: Vec, B: Vec, C: Vec) -> float:
    """cotangent of the angle $\\hat{ABC}$
    
    Args:
        A (Vec): point A
        B (Vec): point B
        C (Vec): point C

    Returns:
        float: the cotangent
    """
    BA = A-B
    BC = C-B
    BA /= norm(BA)
    BC /= norm(BC)
    cosine = dot(BA,BC)
    sine = norm(cross(BA,BC))
    return cosine/sine

@njit
def angle_3pts(A: Vec, B: Vec, C: Vec) -> float:
    """Angle ABC between three points

    Args:
        A (Vec): first point
        B (Vec): central point
        C (Vec): second point

    Returns:
        float: the angle
    """
    BA = A-B
    BC = C-B
    s = np.linalg.norm(cross(BA,BC))
    c = dot(BA,BC)
    return math.atan2(s, c)

@njit
def signed_angle_2vec3D(V1: Vec, V2: Vec, N: Vec) -> float:
    """Signed angle between two vectors with orientation given by normal N

    Args:
        V1 (Vec): First vector
        V2 (Vec): Second vector
        N (Vec): reference normal direction

    Returns:
        float: the angle
    """
    S = cross(V1,V2)
    s = np.linalg.norm(S)
    c = dot(V1,V2)
    return sign0(dot(S,N)) * math.atan2(s, c)

@njit
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

@njit
def angle_2vec2D(V1:Vec, V2:Vec) -> float:
    """Angle between two **2D** vectors

    Args:
        V1 (Vec): first vector
        V2 (Vec): second vector

    Returns:
        float: the angle
    """
    return math.atan2(V2[1], V2[0]) - math.atan2(V1[1], V1[0])

@njit
def angle_2vec3D(V1:Vec, V2:Vec) -> float:
    """Angle between two **3D** vectors

    Args:
        V1 (Vec): first vector
        V2 (Vec): second vector

    Returns:
        float: the angle
    """
    C = cross(V1,V2)
    return math.atan2(norm(C), dot(V1,V2))

def face_basis(*f):
    """
    Orthonormal basis of face
    Given three points A,B,C, returns a basis such that the first vector is along direction AB and third vector is normal to the plane ABC

    Args:
        *f: points of the face, either on a list or not

    Returns:
        Vec,Vec,Vec: an orthonormal 3D basis of the face
    """
    if len(f)==1: f = f[0]
    pA,pB,pC = (x for x in f)
    X = Vec.normalized(pB-pA)
    Z = Vec.normalized(cross(X, pC-pA))
    Y = Vec.normalized(cross(Z,X))
    return X,Y,Z

@njit(inline="always")
def triangle_area(A: Vec, B: Vec, C: Vec) -> float:
    """Area of 3D triangle ABC

    Args:
        A (Vec): first point
        B (Vec): second point
        C (Vec): third point

    Returns:
        float: area
    """
    return norm(cross(B-A,C-A))/2

@njit(inline="always")
def triangle_area_2D(A: Vec, B: Vec, C: Vec) -> float:
    """Area of 2D triangle ABC

    Args:
        A (Vec): first point
        B (Vec): second point
        C (Vec): third point

    Returns:
        float: area
    """
    return abs(det_2x2(B-A,C-A)/2)

@njit
def quad_area(A:Vec, B:Vec, C:Vec, D:Vec) -> float:
    """Area of the quad ABCD

    Args:
        A (Vec): first point
        B (Vec): second point
        C (Vec): third point
        D (Vec): fourth point

    Returns:
        float: area
    """
    return (triangle_area(A,B,C) + triangle_area(A,C,D) + triangle_area(B,C,D) + triangle_area(B,D,A))/2

@njit
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
        args: Either a 3x3 numpy array representing a matrix, or 3 3x1 numpy array representing three column vectors

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

@njit
def intersect_2lines2D(p1 : Vec, d1 : Vec, p2: Vec, d2 : Vec) -> Vec:
    """Computes the intersection of two lines in the plane

    Args:
        p1 (Vec): point on the first line
        d1 (Vec): direction vector of the first line
        p2 (Vec): point on the second line
        d2 (Vec): direction vector of the second line

    Returns:
        Vec: the intersection point. None if lines are parallel
    """
    p1,d1,p2,d2 = p1[:2], d1[:2], p2[:2], d2[:2]
    if abs(det_2x2(d1,d2))<1e-12 : return None #parallel lines
    n2 = np.array((d2[1], -d2[0]))
    t = dot(p2-p1,n2)/dot(d1,n2)
    return p1+t*d1


def circumcenter(v1 : Vec, v2 : Vec, v3: Vec) -> Vec:
    """Circumcenter of the triangle formed by three points

    Parameters:
        v1 (Vec): first point
        v2 (Vec): second point
        v3 (Vec): third point

    Warning:
        Circumcenter of triangle (v1,v2,v3) may not lay inside the triangle

    Returns:
        Vec: coordinates of the circumcenter
    """
    X,Y,_ = face_basis(v1,v2,v3)
    v1,v2,v3 = (np.array((dot(X,v), dot(Y,v))) for v in (v1,v2,v3)) # projet into basis of the triangle
    p1 = (v1+v2)/2
    p2 = (v1+v3)/2
    d1 = v2-v1
    d2 = v3-v1
    d1 = np.array((d1[1], -d1[0]))
    d2 = np.array((d2[1], -d2[0]))
    S = intersect_2lines2D(p1, d1, p2, d2)
    return X*S[0] + Y*S[1]

@njit
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

    References:
        [https://stackoverflow.com/a/10290011](https://stackoverflow.com/a/10290011)
    """
    ab = distance(A,B)
    bc = distance(B,C)
    ca = distance(C,A)
    s = (ab+bc+ca)/2
    return ab*bc*ca/(8*(s-ab)*(s-bc)*(s-ca))


@njit(inline="always", fast_math=True)
def closest_point_on_segment(P: Vec, A: Vec, B: Vec):
    """
    Computes the closest point from p on segment [A;B]
    
    Args:
        P (Vec): Query point
        A (Vec): First segment extremity
        B (Vec): Second segment extremity

    Returns:
        Vec: the point Q on [A;B] that minimizes its distance with P.
    """
    ab = B - A
    l2 = dot(ab, ab)
    t = 0.
    if l2 > 0.:
        t = min(max(dot(P - A, ab) / l2, 0.), 1.)
    return A + t * ab


@njit
def distance_to_segment(P, A, B):
    """
    Computes the distance of point P to the segment [A;B]

    Args:
        P (Vec): Query point
        A (Vec): First segment extremity
        B (Vec): Second segment extremity

    Returns:
        float: the euclidean distance from P to [A;B]
    """
    return distance(P, closest_point_on_segment(P, A, B))



@njit
def project_to_plane(P : Vec, N : Vec, orig : Vec) -> Vec:
    """Projects vector P onto the plane of normal N and passing through point 'orig'

    Args:
        P (Vec): query position to project
        N (Vec): normal vector of the plane

    Returns:
        Vec: P projected onto the plane
    """
    return P - dot(P-orig,N)/dot(N,N)*N