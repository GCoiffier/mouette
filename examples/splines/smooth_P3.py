import mouette as M
import sys

input_file = sys.argv[1]
mesh = M.mesh.load(input_file)

p3 = M.splines.P3Triangulation.from_P1_mesh(mesh, curve=True)
M.mesh.save(p3.rasterize(res=30), "smooth.obj")