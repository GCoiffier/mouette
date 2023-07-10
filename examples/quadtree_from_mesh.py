import mouette as M
import sys
from time import time


input_mesh = M.mesh.load(sys.argv[1])

t0 = time()
domain = M.geometry.BB2D.of_mesh(input_mesh, padding=0.1)
qt = M.processing.QuadTree(domain)

# for v in input_mesh.vertices:
#     qt.insert_point(v.x, v.y)

for v in input_mesh.boundary_vertices:
    p = input_mesh.vertices[v]
    qt.insert_point(p.x, p.y)

print(f"Done in {time()-t0} s")
qt_mesh = qt.export_as_polyline()
M.mesh.save(input_mesh, "input.mesh")
M.mesh.save(qt_mesh, "quadtree.mesh")