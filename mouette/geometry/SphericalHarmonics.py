"""
Attributes:
    EYE (L4_SH): Coefficient of the identity frame in L4_SH
    LX (L4_Operator): Angular momentum operator for spherical harmonics representation, x-axis
    LY (L4_Operator): Angular momentum operator for spherical harmonics representation, y-axis
    LZ (L4_Operator): Angular momentum operator for spherical harmonics representation, z-axis

    Rxpi4 (L4_Operator): rotation of angle pi/4 around axis x
    Rypi4 (L4_Operator): rotation of angle pi/4 around axis y
    Rzpi4 (L4_Operator): rotation of angle pi/4 around axis z
    Rxzpi4 (L4_Operator): composition Rzpi4 * Rxpi4
    Rxpi2 (L4_Operator): rotation of angle pi/2 around axis x
    Rypi2 (L4_Operator): rotation of angle pi/2 around axis y
    Rzpi2 (L4_Operator): rotation of angle pi/2 around axis z

References:
    Algebraic Representations for Volumetric Frame Fields, Palmer et al. (Supplementary materials)
"""

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from math import cos, sin, sqrt, tan, atan
from .vector import Vec
from .geometry import norm, dot
from .rotations import axis_rot_from_z

from typing import Annotated

L4_SH = Annotated[np.ndarray, 9]
"""
Type annotation for linear combinations of L4 Spherical Harmonics. 
Orthogonal frames are represented by 9 coefficients in the L4 band of the spherical harmonics basis. 
In practice, L4_SH is a size 9 numpy array.
"""

L4_Operator = Annotated[np.ndarray, (9,9)]
"""
Type annotation for linear operators acting on L4_SH (i.e. 9x9 matrices)
"""

LX = np.array([
    [0.,      0.,        0.,        0.,        0.,        0.,         0.,         -sqrt(2),   0.      ],
    [0.,      0.,        0.,        0.,        0.,        0.,         -sqrt(7/2), 0.,         -sqrt(2)],
    [0.,      0.,        0.,        0.,        0.,        -3/sqrt(2), 0.,         -sqrt(7/2), 0.      ],
    [0.,      0.,        0.,        0.,        -sqrt(10), 0.,         -3/sqrt(2), 0.,         0.      ],
    [0.,      0.,        0.,        sqrt(10),  0.,        0.,         0.,         0.,         0.      ],
    [0.,      0.,        3/sqrt(2), 0.,        0.,        0.,         0.,         0.,         0.      ],
    [0.,      sqrt(7/2), 0.,        3/sqrt(2), 0.,        0.,         0.,         0.,         0.      ],
    [sqrt(2), 0.,        sqrt(7/2), 0.,        0.,        0.,         0.,         0.,         0.      ],
    [0.,      sqrt(2),   0.,        0.,        0.,        0.,         0.,         0.,         0.      ],
])
# Angular momentum operator for spherical harmonics representation, x-axis


LY = np.array([
    [0.,       sqrt(2),    0.,         0.,        0.,       0.,        0.,         0.,         0.      ],
    [-sqrt(2), 0.,         sqrt(7/2),  0.,        0.,       0.,        0.,         0.,         0.      ],
    [0.,       -sqrt(7/2), 0.,         3/sqrt(2), 0.,       0.,        0.,         0.,         0.      ],
    [0.,       0.,         -3/sqrt(2), 0.,        0.,       0.,        0.,         0.,         0.      ],
    [0.,       0.,         0.,         0.,        0.,       -sqrt(10), 0.,         0.,         0.      ],
    [0.,       0.,         0.,         0.,        sqrt(10), 0.,        -3/sqrt(2), 0.,         0.      ],
    [0.,       0.,         0.,         0.,        0.,       3/sqrt(2), 0.,         -sqrt(7/2), 0.      ],
    [0.,       0.,         0.,         0.,        0.,       0.,        sqrt(7/2),  0.,         -sqrt(2)],
    [0.,       0.,         0.,         0.,        0.,       0.,        0.,         sqrt(2),    0.      ],
])
# Angular momentum operators for spherical harmonics representation, y-axis


LZ = np.array([
    [0.,  0.,  0.,  0., 0., 0., 0., 0., 4.],
    [0.,  0.,  0.,  0., 0., 0., 0., 3., 0.],
    [0.,  0.,  0.,  0., 0., 0., 2., 0., 0.],
    [0.,  0.,  0.,  0., 0., 1., 0., 0., 0.],
    [0.,  0.,  0.,  0., 0., 0., 0., 0., 0.],
    [0.,  0.,  0., -1., 0., 0., 0., 0., 0.],
    [0.,  0., -2.,  0., 0., 0., 0., 0., 0.],
    [0., -3.,  0.,  0., 0., 0., 0., 0., 0.],
    [-4., 0.,  0.,  0., 0., 0., 0., 0., 0.],
])
# Angular momentum operators for spherical harmonics representation, z-axis

Rxpi4  : L4_Operator = expm(np.pi/4 * LX)
Rypi4  : L4_Operator = expm(np.pi/4 * LY)
Rzpi4  : L4_Operator = expm(np.pi/4 * LZ)
Rxzpi4 : L4_Operator = Rzpi4 @ Rxpi4
Rxpi2  : L4_Operator = expm(np.pi/2 * LX)
Rypi2  : L4_Operator = expm(np.pi/2 * LY)
Rzpi4  : L4_Operator = expm(np.pi/2 * LZ)
# Hard-coded rotation matrices for common angles

def RZ(a : float) -> L4_Operator:
    """
    The L4 operator representing a rotation around z-axis of angle a

    Args:
        a (float): angle

    Returns:
        L4_Operator : 9x9 matrix
    """
    c = [cos(k*a) for k in range(5)]
    s = [sin(k*a) for k in range(5)]
    return np.array([
        [ c[4], 0.,   0.,   0.,   0., 0.,   0.,   0.,   s[4]],
        [ 0.,   c[3], 0.,   0.,   0., 0.,   0.,   s[3], 0.  ],
        [ 0.,   0.,   c[2], 0.,   0., 0.,   s[2], 0.,   0.  ],
        [ 0.,   0.,   0.,   c[1], 0., s[1], 0.,   0.,   0.  ],
        [ 0.,   0.,   0.,   0.,   1., 0.,   0.,   0.,   0.  ],
        [ 0.,   0.,   0.,  -s[1], 0., c[1], 0.,   0.,   0.  ],
        [ 0.,   0.,  -s[2], 0.,   0., 0.,   c[2], 0.,   0.  ],
        [ 0.,  -s[3], 0.,   0.,   0., 0.,   0.,   c[3], 0.  ],
        [-s[4], 0.,   0.,   0.,   0., 0.,   0.,   0.,   c[4]],
    ])

def RY(a : float) -> L4_Operator:
    """
    The L4 operator representing a rotation around y-axis of angle a

    Args:
        a (float): angle

    Returns:
        L4_Operator : 9x9 matrix
    """
    return Rxpi2.T @ RZ(a) @ Rxpi2

def RX(a : float) -> L4_Operator:
    """
    The L4 operator representing a rotation around x-axis of angle a

    Args:
        a (float): angle

    Returns:
        L4_Operator : 9x9 matrix
    """
    return Rypi2 @ RZ(a) @ Rypi2.T


EYE : L4_SH = np.array([0., 0., 0., 0., sqrt(7/12), 0., 0., 0., sqrt(5/12) ])
# Coefficient of the identity frame in L4_SH

def skew_matrix_from_rotvec(w : Vec) -> L4_Operator :
    """
    Given a rotation in angle-axis form, computes the 9x9 corresponding skew-symmetric matrix
    
    Args:
        w (Vec): a rotation axis vector where norm(v) represents the angle of rotation.

    Returns:
        L4_Operator: a 9x9 matrix
    """
    return w[0] * LX + w[1] * LY + w[2] * LZ

def rot_matrix_from_euler(w : Vec) -> L4_Operator :
    """
    Given three euler angles (XYZ), computes the 9x9 corresponding rotation matrix that performs the same rotation onto L4_SH coefficients

    Args:
        w (Vec): vector of size 3 representing three euler angles in convention XYZ.

    Returns:
        L4_Operator: a 9x9 matrix
    """
    return RX(w[0]) @ RY(w[1]) @ RZ(w[2])

def rot_matrix_from_rotvec(v : Vec) -> L4_Operator :
    """
    Given a rotation in angle-axis form, computes the 9x9 corresponding rotation matrix that performs the same rotation onto L4_SH coefficients,
    defined as the exponential of the corresponding skew-symmetric matrix.

    Args:
        v (Vec): a rotation axis vector where norm(v) represents the angle of rotation.

    Returns:
        L4_Operator: a 9x9 matrix
    """
    return expm(skew_matrix_from_rotvec(v))

def from_vec3(v : Vec) -> L4_SH:
    """
    Given a rotation in angle-axis form, computes the corresponding frame coefficients in L4_SH basis.

    Args:
        v (Vec): a rotation axis vector where norm(v) represents the angle of rotation.

    Returns:
        L4_SH: corresponding coefficients
    """
    if norm(v)<1e-8:  return np.array([0.,0.,0.,0.,1.,0.,0.,0.,0.])
    R = rot_matrix_from_rotvec(v)
    sh = R[:,4] # apply rotation to representation of (0,0,1) vector : 1 at 4th coeff and 0 otherwise -> dot product is fourth column of matrix
    return sh

def from_frame(frame : Rotation) -> L4_SH:
    """
    Given a rotation as a scipy.Rotation object, computes the corresponding frame coefficients in L4_SH basis.

    Args:
        frame (Rotation): a scipy rotation.

    Returns:
        L4_SH: corresponding coefficients
    """
    axis = Vec(frame.as_euler("XYZ"))
    if norm(axis)<1e-8: return EYE
    R = rot_matrix_from_euler(axis)
    sh = R.dot(EYE) # apply found rotation to the spherical harmonic reference
    return sh

def rotate_frame(sh : L4_SH, r : Rotation) -> L4_SH :
    """
    Applies a rotation to a frame decomposed in L4_SH basis.

    Args:
        sh (L4_SH): coefficients of the frame
        r (Rotation|Vec): rotation to be applied, either a scipy.Rotation object or a axis-angle representation.

    Returns:
        L4_SH : the rotated frame coefficients
    """
    if isinstance(r, Rotation):
        return Vec(rot_matrix_from_euler(r.as_rotvec()).dot(sh))
    return Vec(rot_matrix_from_euler(r).dot(sh))

def project_to_frame(sh : Vec, stop_threshold : float = 1e-8, max_iter=1000, nrml_cstr: Vec = None):
    """Given the coefficients in the spherical harmonic basis, finds the frames that correspond the most.
    Also recomputes spherical harmonics coefficients to a perfect match.
    Uses the Cayleigh transform to approximate the 9D rotation.

    Parameters:
        sh (Vec): the 9 coefficients representing the frame in spherical harmonics basis
        stop_threshold (float,optional) : Stoping criterion on the norm of the residual. Defaults to 1e-8.
        max_iter (int, optional): Maximum number of gradient steps. Defaults to 100.
        nrml_cstr (Vec) : a 3 dimensionnal normal direction, to constraint the projection to only be around an axis. Defaults to None.
    
    Returns:
        frame (scipy.spatial.transform.Rotation) : the obtained frame
        a (L4_SH) : the projected and updated spherical harmonics coefficients
    """
    if norm(sh)<1e-6: return Rotation.identity(), sh
    q = Vec.normalized(sh)
    
    if nrml_cstr is None:
        seeds = [EYE] + [R.dot(EYE) for R in (Rxpi4, Rypi4, Rzpi4, Rxzpi4)]
        pi4 = np.pi/4
        frames = [Rotation.identity(),
                Rotation.from_euler("x", pi4),
                Rotation.from_euler("y", pi4),
                Rotation.from_euler("z", pi4),
                Rotation.from_euler("xz", (pi4,pi4))]
        dots = [ np.dot(q, seed) for seed in seeds]
        seed_choice = np.argmax(dots)
        a = seeds[seed_choice]
        frame = frames[seed_choice]
        for _ in range(max_iter):
            s, b = (q+a)/2, (q-a)
            l = np.array([ LX.dot(s), LY.dot(s), LZ.dot(s)])
            if norm( l.dot(b))<stop_threshold: break
            x = np.linalg.lstsq(l.T, b, rcond=None)[0]
            x = Vec.normalized(x) * 2*atan(norm(x)/2)
            rot = Rotation.from_rotvec(x)
            frame = rot * frame
            x = rot.as_euler("XYZ")
            a = rot_matrix_from_euler(x).dot(a)
    else:
        ax = axis_rot_from_z(nrml_cstr)
        frames = [ R * Rotation.from_rotvec(ax) for R in 
                    (Rotation.identity(),
                     Rotation.from_rotvec(np.pi/4 * nrml_cstr))] 
        seeds = [from_frame(_frame) for _frame in frames]
        dots = [ np.dot(q, seed) for seed in seeds]
        seed_choice = np.argmax(dots)
        a = seeds[seed_choice]
        frame = frames[seed_choice]
        L = skew_matrix_from_rotvec(nrml_cstr)
        for _ in range(max_iter):
            s,b = L.dot(q+a), q-a
            if abs(s.dot(b))<stop_threshold: break
            alpha = dot(s,b)/dot(s,s)
            rot = Rotation.from_rotvec(2*atan(alpha)*nrml_cstr)
            frame = rot * frame
            x = rot.as_euler("XYZ")
            a = rot_matrix_from_euler(x).dot(a)
    return frame,a

def project_to_frame_grad(sh : Vec, lr : float = 1e-1, grad_threshold : float = 1e-4, dot_threshold = 1e-6, max_iter=1000):
    """Given the coefficients in the spherical harmonic basis, finds the frames that correspond the most. Computation is unfortunately not direct:
    starting from the reference frame, we perform a gradient descent on the l2 distance between spherical harmonics coefficients.

    Also recomputes spherical harmonics coefficients to a perfect match.
    Uses the linearization of the exponential to approximate 9D rotations.
    
    Warning:
        This algorithm is less precise and less efficient than `project_to_frame`. Use the latter instead.

    Parameters:
        sh (Vec): the 9 coefficients representing the frame in spherical harmonics basis
        lr (float, optional): Gradient descent learning rate. Defaults to 1.
        grad_threshold (float,optional) : Stoping criterion on the norm of the gradient. Defaults to 1e-4
        dot_threshold (float,optional) : Stoping criterion on the norm of the step. Defaults to 1e-6
        max_iter (int, optional): Maximum number of gradient steps. Defaults to 1000.
    
    Returns:
        frame (scipy.spatial.transform.Rotation) : the obtained frame
        a (Vec) : the projected and updated spherical harmonics coefficients
    """

    if norm(sh)<1e-6: return Rotation.identity(), sh
    Id = np.eye(9)
    q = Vec.normalized(sh)
    Rx, Ry, Rz = (Id + np.pi/4 * L for L in (LX, LY, LZ))
    qLX, qLY, qLZ = (np.dot(L.T, q) for L in (LX, LY, LZ))
    seeds = [EYE] + [R.dot(EYE) for R in (Rxpi4, Rypi4, Rzpi4, Rxzpi4)]
    pi4 = np.pi/4
    frames = [Rotation.identity(),
              Rotation.from_euler("x", pi4),
              Rotation.from_euler("y", pi4),
              Rotation.from_euler("z", pi4),
              Rotation.from_euler("xz", (pi4,pi4))]
    dots = [ np.dot(q, seed) for seed in seeds]
    seed_choice = np.argmax(dots)
    old_dot = dots[seed_choice]

    a = seeds[seed_choice]
    frame = frames[seed_choice]
    n_iter = 0
    while n_iter<max_iter:
        gx,gy,gz = ( np.dot(_q, a) for _q in (qLX,qLY,qLZ))
        grad = Vec( gx, gy, gz ) # gradient in point a
        if grad.norm()< grad_threshold: break

        step = lr*grad
        # technically, Rx, Ry and Rz should be exponentials of L1, L2, L3 with corresponding angles,
        # but we approximate at first order exp(X) = 1 + X + o(|X|)
        Rx = Id + step.x * LX
        Ry = Id + step.y * LY
        Rz = Id + step.z * LZ
        
        R = Rx @ Ry @ Rz
        a = R.dot(a)
        
        new_dot = np.dot(q,a)
        if abs(old_dot-new_dot)<dot_threshold: break
        old_dot = new_dot

        frame = Rotation.from_euler("XYZ", [step.x, step.y, step.z]) * frame
        n_iter += 1
    return frame,a

def orthogonality_energy(sh : np.ndarray) -> float:
    """Computes an energy that is 0 if and only if the given spherical harmonics has octahedral symetry (ie represents an orthogonal frame)
    The energy is quadratic in terms of the harmonic coefficients

    Parameters:
        sh (np.ndarray): a spherical harmonics (9 coefficients)

    Returns:
        (float): the computed energy
    """
    s1 = 28*(sh[0]*sh[0] + sh[8]*sh[8]) + 7*(sh[1]*sh[1] + sh[7]*sh[7]) - 8*(sh[2]*sh[2] + sh[6]*sh[6]) - 17*(sh[3]*sh[3] + sh[5]*sh[5]) - 20*sh[4]*sh[4]
    s1 *= sqrt(2)

    s2 = 28*(sh[0]*sh[1] + sh[7]*sh[8]) + 10*sqrt(7)*(sh[1]*sh[2] + sh[6]*sh[7]) + 18*(sh[2]*sh[3] + sh[5]*sh[6]) + 4*sqrt(5)*sh[4]*sh[5]
    s2 *= sqrt(3)

    s3 = 28*(sh[0]*sh[7] - sh[1]*sh[8]) + 10*sqrt(7)*(sh[1]*sh[6] - sh[2]*sh[7]) + 4*sqrt(5)*sh[4]*sh[3] + 18*(sh[2]*sh[5] - sh[3]*sh[6])
    s3 *= sqrt(3)

    s4 = 4*sqrt(7)*(sh[0]*sh[2] + sh[6]*sh[8]) + 6*sqrt(7)*(sh[1]*sh[3] + sh[5]*sh[7]) + 12*sqrt(5)*sh[4]*sh[6] + 10*(sh[3]*sh[3] - sh[5]*sh[5])
    s4 *= sqrt(6)

    s5 = 4*sqrt(7)*(sh[6]*sh[0] - sh[2]*sh[8]) + 6*sqrt(7)*(sh[1]*sh[5] - sh[3]*sh[7]) + 12*sqrt(5)*sh[2]*sh[4] - 20*sh[3]*sh[5]
    s5 *= sqrt(6)

    e = np.array((s1,s2,s3,s4,s5))
    return np.dot(e,e)