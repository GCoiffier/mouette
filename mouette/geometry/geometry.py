from .vector import Vec
from ..utils import check_argument
import numpy as np
import math
from typing import Union

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

def norm(x: np.ndarray, which="l2") -> float:
    """Vector norm. Three norms are implemented, the Euclidean l2 norm, the l1 norm or the l-infinite norm:

    l2: $ \\sqrt{ \\sum_i v[i]^2 } $

    l1: $ \\sum_i |v[i]| $

    linf: $ \\max_i |v[i]| $

    Args:
        x (np.ndarray): vector from which we compute the norm. Will be flattened if it has more than one dimension.
        which (str, optional): which norm to compute. Choices are "l2", "l1" and "linf". Defaults to "l2".

    Returns:
        float: the vector's norm
    """
    check_argument("which", which, str, ["l2", "l1", "linf"])
    if which=="l2":
        return np.sqrt(np.dot(x.flatten(), x.flatten()))
    elif which=="l1":
        return np.sum(np.abs(x))
    elif which=="linf":
        return np.max(np.abs(x))

def dot(A: np.ndarray, B: np.ndarray) -> float:
    """dot product $A^TB$ between two vectors

    Args:
        A (np.ndarray): first vector
        B (np.ndarray): second vector

    Returns:
        float: dot product
    """
    return np.dot(A,B)

def distance(A: Vec, B: Vec, which="l2") -> float:
    """Distance between A and B. Three distances are implemented, the Euclidean distance l2, the Manhattan l1 distance of the l-infinity distance:
     
    l2: $\\sqrt{(B-A)^T(B-A)} = \\sqrt{\\sum_i (A_i - B_i)^2}$

    l1: $\\sum_i |A_i - B_i|$

    linf: $\\max_i |A_i - B_i|$

    Args:
        A (Vec): first point
        B (Vec): second point
        which (str, optional): which norm to compute. Choices are "l2", "l1" and "linf". Defaults to "l2".
        
    Returns:
        float: distance
    """
    return norm(B-A, which)

def cross(A: Vec, B: Vec) -> Vec:
    """Cross product of vectors A and B

    Args:
        A (Vec): Size 3 vector
        B (Vec): Size 3 vector
        
    Returns:
        Vec: cross product AxB
    """
    return Vec(
        A[1]*B[2] - A[2] * B[1],
        B[0]*A[2] - B[2] * A[0],
        A[0]*B[1] - A[1] * B[0]
    )

def cotan(A: Vec, B: Vec, C: Vec) -> float:
    """cotangent of the angle $\\hat{ABC}$
    Parameters:
        A (Vec): point A
        B (Vec): point B
        C (Vec): point C

    Returns:
        float: the cotangent
    """
    A, B, C = Vec(A), Vec(B), Vec(C)
    BA = Vec.normalized(A-B)
    BC = Vec.normalized(C-B)
    cosine = np.dot(BA,BC)
    sine = norm(cross(BA,BC))
    return cosine/sine

def angle_3pts(A: Vec, B: Vec, C: Vec) -> float:
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


def signed_angle_2vec3D(V1: Vec, V2: Vec, N: Vec) -> float:
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
    """Angle between two **2D** vectors

    Args:
        V1 (Vec): first vector
        V2 (Vec): second vector

    Returns:
        float: the angle
    """
    return math.atan2(V2.y, V2.x) - math.atan2(V1.y, V1.x)

def angle_2vec3D(V1:Vec, V2:Vec) -> float:
    """Angle between two **3D** vectors

    Args:
        V1 (Vec): first vector
        V2 (Vec): second vector

    Returns:
        float: the angle
    """
    C = cross(V1,V2)
    return math.atan2(C.norm(), dot(V1,V2))

def face_basis(*f):
    """
    Orthonormal basis of face
    Given three points A,B,C, returns a basis such that the first vector is along direction AB and third vector is normal to the plane ABC

    Args:
        A,B,C: three points (in a list or not)

    Returns:
        Vec,Vec,Vec: an orthonormal 3D basis of the face
    """
    if len(f)==1: f = f[0]
    pA,pB,pC = (x for x in f)
    X = Vec.normalized(pB-pA)
    Z = Vec.normalized(cross(X, pC-pA))
    Y = Vec.normalized(cross(Z,X))
    return X,Y,Z

def triangle_area(A: Vec, B: Vec, C: Vec) -> float:
    """Area of 3D triangle ABC

    Args:
        A (Vec): first point
        B (Vec): second point
        C (Vec): third point

    Returns:
        float: area
    """
    return cross(B-A,C-A).norm()/2

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
    p1,d1,p2,d2 = (u[:2] for u in (p1,d1,p2,d2))
    if abs(det_2x2(d1,d2))<1e-12 : return None #parallel lines
    n2 = Vec(d2.y, -d2.x)
    t = dot(p2-p1,n2)/dot(d1,n2)
    return p1+t*d1

def circumcenter(v1 : Vec, v2 : Vec, v3: Vec) -> Vec:
    """Circumcenter of the triangle formed by three points in space

    Parameters:
        v1 (Vec): first point
        v2 (Vec): second point
        v3 (Vec): third point

    Warning:
        Circumcenter of triangle (v1,v2,v3) may not lay inside the triangle

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
        [https://stackoverflow.com/a/10290011](https://stackoverflow.com/a/10290011)
    """
    ab = distance(A,B)
    bc = distance(B,C)
    ca = distance(C,A)
    s = (ab+bc+ca)/2
    return ab*bc*ca/(8*(s-ab)*(s-bc)*(s-ca))


def distance_to_segment2D(P : Vec, A : Vec, B : Vec) -> float:
    """
    Computes, **in the plane**, the distance of point P to the segment [A;B]

    Args:
        P (Vec): Query point in 2D
        A (Vec): First segment extremity
        B (Vec): Second segment extremity

    Returns:
        float: the euclidean distance from P to [A;B]
    """
    P,A,B = P[:2], A[:2], B[:2]
    seg = B-A
    seg_length_sq = dot(seg,seg)
    if seg_length_sq<1e-12: 
        # segment is a single point
        return distance(P,A)
    t = max(0, min(1, dot(P-A, seg)/seg_length_sq))
    proj = A + t*seg
    return distance(P, proj)


def project_to_plane(P : Vec, N : Vec, orig : Vec) -> Vec:
    """Projects vector P onto the plane of normal N and passing through point 'orig'

    Args:
        P (Vec): query position to project
        N (Vec): normal vector of the plane

    Returns:
        Vec: P projected onto the plane
    """
    return P - dot(P-orig,N)/dot(N,N)*N