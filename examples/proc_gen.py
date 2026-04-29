import mouette as M
from mouette import procedural as Mproc

M.mesh.save(
    Mproc.tetrahedron(M.Vec(1,1,1), M.Vec(1,-1,-1), M.Vec(-1,1,-1), M.Vec(-1,-1,1)), "shape_tetrahedron.obj")

M.mesh.save(Mproc.axis_aligned_cube(), "shape_cube.obj")

M.mesh.save(Mproc.octahedron(), "shape_octahedron.obj")

M.mesh.save(Mproc.icosahedron(), "shape_icosahedron.obj")

M.mesh.save(Mproc.dodecahedron(), "shape_dodecahedron.obj")

M.mesh.save(Mproc.cylinder(M.Vec(0,0,0), M.Vec(1,1,1)), "shape_cylinder.obj")

M.mesh.save(Mproc.sphere_uv(30,50), "shape_sphere.obj")

M.mesh.save(Mproc.torus(50,30, 3., 1.), "shape_torus.obj")