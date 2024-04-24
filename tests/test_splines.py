import mouette as M
import numpy as np

def test_generate_bezier_curve():
    P0 = M.Vec.random(3)
    P1 = M.Vec.random(3)
    P2 = M.Vec.random(3)
    P3 = M.Vec.random(3)
    pts = np.array((P0,P1,P2,P3))
    curv = M.splines.BezierCurve(pts)
    assert curv.order == 3
    polyline = curv.as_polyline(n_pts=100)
    assert len(polyline.vertices) == 100

def test_generate_bezier_patch():
    Nu,Nv = 4,4
    U = np.linspace(-1,1,Nu)
    V = np.linspace(-1,1,Nv)
    z = lambda u,v : u*u + 2*u*v + v
    control_points = []
    for i in range(Nu):
        pts_u = []
        for j in range(Nv):
            pt = M.Vec(U[i], V[j], 0.3*z(U[i], V[j]))
            pts_u.append(pt)
        control_points.append(pts_u)
    surf = M.splines.BezierPatch(control_points)
    assert surf.order == (3,3)
    mesh = surf.as_surface(40, 40)
    assert len(mesh.vertices)==40*40
    assert len(mesh.faces)==39*39