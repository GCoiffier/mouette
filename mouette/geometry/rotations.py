from .vector import Vec
from scipy.spatial.transform import Rotation
import math
from .. import geometry as geom

def rotate_2d(v : Vec, angle : float) -> Vec:
    """Rotates a 2D vector in the plane

    Parameters:
        v (Vec): the vector to rotate
        angle (float): angle of rotation in radiants

    Returns:
        Vec: the rotated vector
    """
    ca,sa = math.cos(angle), math.sin(angle)
    v2 = Vec(0.,0.)
    v2.x = v.x * ca - v.y * sa
    v2.y = v.x * sa + v.y * ca
    return v2

def rotate_around_axis(inp : Vec, _axis : Vec, angle : float) ->  Vec:
    c,s = math.cos(angle), math.sin(angle)
    inp = Vec(inp)
    axis = Vec.normalized(_axis)
    if abs(angle)<1E-12 or axis.norm()<1E-12: return inp
    out = Vec(0.,0.,0.)
    u,v,w = axis
    out.x = (c + u*u*(1-c))   * inp.x + (u*v*(1-c) - w*s) * inp.y + (u*w*(1-c) + v*s) * inp.z
    out.y = (u*v*(1-c) + w*s) * inp.x + (c + v*v*(1-c))   * inp.y + (v*w*(1-c) - u*s) * inp.z
    out.z = (u*w*(1-c) - v*s) * inp.x + (v*w*(1-c) + u*s) * inp.y + (c + w*w*(1-c))   * inp.z
    return out

def axis_rot_from_z(v : Vec) -> Vec:
    """The (smallest) rotation to align the z axis (0,0,1) with the vector v
    """
    Z = Vec(0.,0.,1.)
    axis = geom.cross(Z, v)
    angle = geom.angle_2vec3D(Z, v)
    if axis.norm()>1e-8: axis = Vec.normalized(axis) * angle
    return axis

def match_rotation(Ra : Rotation, Rb : Rotation, symgroup = Rotation.create_group("O"), threshold=math.pi/4):
    """Given two rotation matrices, compute the matching between the two: 
    the minimal rotation going from Ra to s(Rb) with s in some symmetry group (most often the octahedral group for cube symmetries)

    Parameters:
        Ra (Rotation)
        Rb (Rotation)
        symgroup (Rotation) : a rotation group (given by the `scipy.spatial.transform.Rotation.create_group` method). 
            Defaults to the octahedral group (24 direct symmetries of the cube)

    Returns:
        Rotation: the minimal rotation going from Ra to Rb
    """
    RaT = Ra.inv()
    Rab = Rb * RaT
    if Rab.magnitude()<threshold: return Rab # no need to perform a symmetry
    #return Rab.reduce(left=symgroup)
    min_w = float("inf")
    min_r = None
    for S in symgroup:
        match = Rb * S * RaT
        w = match.magnitude()
        if w<threshold: return match
        if w<min_w:
            min_w = w
            min_r = match
    return min_r