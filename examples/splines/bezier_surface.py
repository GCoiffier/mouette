import mouette as M
import numpy as np

Nu,Nv = 4,4

U = np.linspace(-1,1,Nu)
V = np.linspace(-1,1,Nv)

def z(u,v):
    return u*u + 2*u*v + v

control_pc = M.mesh.RawMeshData()
control_points = []
for i in range(Nu):
    pts_u = []
    for j in range(Nv):
        pt = M.Vec(U[i], V[j], 0.3*z(U[i], V[j]))
        pts_u.append(pt)
        control_pc.vertices.append(pt)
    control_points.append(pts_u)
control_pc = control_pc.instantiate(0)

surf = M.splines.BezierPatch(control_points)
mesh = surf.as_surface(40, 40)

M.mesh.save(mesh, "bezier.obj")
M.mesh.save(control_pc,"control_points.obj")