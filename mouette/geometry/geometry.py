import numpy as np
from numpy import dot
from numpy.linalg import norm

from numba import njit
import math
from typing import Union, TypeAlias

# A Vec type represents here a 1D numpy array
Vec : TypeAlias = np.ndarray

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

@njit
def face_basis(A: Vec, B: Vec, C: Vec):
    """
    Orthonormal basis of face
    Given three points A,B,C, returns a basis such that the first vector is along direction AB and third vector is normal to the plane ABC

    Args:
        A, B, C (Vec): three points forming a triangle.

    Returns:
        Vec,Vec,Vec: an orthonormal 3D basis of the face ABC
    """
    X = B-A
    X /= norm(X)
    Z = cross(X, C-A)
    Z /= norm(Z)
    Y = cross(Z,X)
    Y /= norm(Y)
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

    Args:
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

@njit
def det_3x3(A: Vec, B: Vec, C:Vec) -> float:
    """Computes a 3x3 determinant using the rule of Sarrus from three column vectors.
    
    Args:
        A (Vec): first column
        B (Vec): second column
        C (Vec): third column

    Returns:
        float: the determinant of [A|B|C]
    """
    return A[0] * B[1] * C[2] + \
           B[0] * C[1] * A[2] + \
           C[0] * A[1] * B[2] - \
           A[0] * C[1] * B[2] - \
           B[0] * A[1] * C[2] - \
           C[0] * B[1] * A[2]

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


@njit
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
    # project into basis of the triangle
    v1, v2, v3 = np.array((dot(X,v1), dot(Y,v1))), np.array((dot(X,v2), dot(Y,v2))), np.array((dot(X,v3), dot(Y,v3)))
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


@njit(inline="always", fastmath=True)
def closest_point_on_segment(P: Vec, A: Vec, B: Vec):
    """
    Computes the closest point from p on segment [A;B], and the feature of the segment on which it lies.
    
    Args:
        P (Vec): Query point
        A (Vec): First segment extremity
        B (Vec): Second segment extremity

    Returns:
        Vec: the point Q on [A;B] that minimizes its distance with P.
        int: the feature of the segment on which Q lies: 0 if Q is in the interior of the segment, 1 if Q is the extremity A, 2 if Q is the extremity B.
    """
    ab = B - A
    l2 = dot(ab, ab)
    t = 0.
    if l2 > 0.:
        t = dot(P - A, ab) / l2
    if t <= 0.: return A.copy(), 1
    if t >= 1.: return B.copy(), 2
    return A + t * ab, 0


@njit
def distance_to_segment(P, A, B):
    """
    Computes the distance of point P to the segment [A;B].
    Alias for `distance(P, closest_point_on_segment(P,A,B)[0])`

    Args:
        P (Vec): Query point
        A (Vec): First segment extremity
        B (Vec): Second segment extremity

    Returns:
        float: the euclidean distance from P to [A;B]
    """
    return distance(P, closest_point_on_segment(P, A, B)[0])


@njit(inline="always", fastmath=True)
def closest_point_on_triangle(P: Vec, A: Vec, B: Vec, C: Vec):
    """
    Computes the closest point from point p to triangle ABC, and the feature of the triangle on which it lies.
    Uses the Voronoi region tests from Ericson, Real-Time Collision Detection, section 5.1.5.

    Args:
        P (Vec): Query point
        A,B,C (Vec): vertices of the triangle

    Returns:
        Vec: the point Q in triangle ABC that minimizes its distance with P.
        int: the feature of the triangle on which Q lies: 
            0 if Q is in the interior of the face, 
            1, 2 or 3 if Q is the vertex A, B or C respectively, 
            4, 5 or 6 if Q is in the interior of the edge AB, BC or CA respectively.
    """
    ab, ac, ap = B - A, C - A, P - A
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    if nx * nx + ny * ny + nz * nz <= 1e-12 * np.dot(ab, ab) * np.dot(ac, ac):
        # degenerate (zero area) triangle: region tests are unreliable, take the closest of the three edges
        q_ab, f_ab = closest_point_on_segment(P, A, B)
        q_bc, f_bc = closest_point_on_segment(P, B, C)
        q_ca, f_ca = closest_point_on_segment(P, C, A)
        d_ab, d_bc, d_ca = distance2(P, q_ab), distance2(P, q_bc), distance2(P, q_ca)
        # convert the segment features (0: interior, 1: first extremity, 2: second extremity) into triangle features
        if d_ab <= d_bc and d_ab <= d_ca: return q_ab, (4, 1, 2)[f_ab]
        elif d_bc <= d_ca: return q_bc, (5, 2, 3)[f_bc]
        else: return q_ca, (6, 3, 1)[f_ca]
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0. and d2 <= 0.: return A.copy(), 1 # vertex region A
    bp, cp = P - B, P - C
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0. and d4 <= d3: return B.copy(), 2 # vertex region B
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0. and d5 <= d6: return C.copy(), 3 # vertex region C
    vc = d1 * d4 - d3 * d2
    if vc <= 0. and d1 >= 0. and d3 <= 0.: # edge region AB
        return A + (d1 / (d1 - d3)) * ab, 4
    vb = d5 * d2 - d1 * d6
    if vb <= 0. and d2 >= 0. and d6 <= 0.: # edge region CA
        return A + (d2 / (d2 - d6)) * ac, 6
    va = d3 * d6 - d5 * d4
    if va <= 0. and d4 >= d3 and d5 >= d6: # edge region BC
        return B + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (C - B), 5
    # inside the face. Denominators above are squared edge lengths and squared double area, all nonzero since the triangle is not degenerate
    denom = va + vb + vc
    return A + (vb / denom) * ab + (vc / denom) * ac, 0


@njit
def distance_to_triangle(P:Vec, A:Vec, B:Vec, C:Vec) -> float:
    """
    Computes the distance of point P to the triangle ABC.
    Alias for `distance(P, closest_point_on_triangle(P,A,B,C)[0])`

    Args:
        P (Vec): query point
        A,B,C (Vec): vertices of the triangle.

    Returns:
        float: float: the euclidean distance from P to triangle ABC.
    """
    return distance(P, closest_point_on_triangle(P, A, B, C)[0])


@njit(inline="always", fastmath=True)
def closest_point_on_plane(P : Vec, orig : Vec, N: Vec) -> Vec:
    """Projects vector P onto the plane of normal N and passing through point 'orig'

    Args:
        P (Vec): query position to project
        orig (Vec): a point belonging to the plane
        N (Vec): normal vector of the plane

    Returns:
        Vec: P projected onto the plane
    """
    return P - dot(P-orig,N)/dot(N,N)*N

@njit(inline="always", fastmath=True)
def distance_to_plane(P: Vec, orig:Vec, N:Vec) -> float:
    """
    Computes the distance of point P to the plane of normal N passing through point `orig`.

    Args:
        P (Vec): query position to project
        orig (Vec): a point belonging to the plane
        N (Vec): normal vector of the plane. Does not need to be normalized.

    Raises:
        ValueError: if the norm of N is smaller than 1e-10.

    Returns:
        float: the euclidean distance from P to the plane.
    """
    nN = norm(N)
    if nN < 1e-10: raise ValueError("distance_to_plane: the normal vector of the plane has a norm smaller than 1e-10")
    return np.abs(dot(P-orig, N)) / nN