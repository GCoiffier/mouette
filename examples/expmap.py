import mouette as M
import sys
from random import randint

if len(sys.argv)<2:
    print("Usage: `python expmap.py <path/to/mesh>")

mesh = M.mesh.load(sys.argv[1])
conn = M.processing.SurfaceConnectionVertices(mesh)

expmap = M.processing.DiscreteExponentialMap(mesh, conn, dist=10)

### Take an origin vertex at random
Vorig = mesh.boundary_vertices[0] # randint(0, len(mesh.vertices)-1)
selected = mesh.vertices.create_attribute("selection", bool)
selected[Vorig] = True

### Export exponential map of this vertex as uv coordinates
uv_coords = mesh.vertices.create_attribute("uv_coords", float, 2, dense=True)
for v in mesh.id_vertices:
    if expmap.map(Vorig, v) is not None:
        uv_coords[v] = expmap.map(Vorig, v)

### Output files
M.mesh.save(mesh, M.utils.get_filename(sys.argv[1])+".geogram_ascii")
M.mesh.save(mesh, M.utils.get_filename(sys.argv[1])+".obj")

map_flat = expmap.export_map_as_mesh(Vorig)
M.mesh.save(map_flat, f"map_{Vorig}.obj")