---
title: "Mesh Subdivision"
---

Due to the connectivity data attached to the meshes data structures, subdividing meshes by hand is unsafe and can lead to faulty results. To correctly apply subdivisions, one needs to first convert back the mesh to a [`RawMeshData`](mouette.mesh.mesh_data.RawMeshData), apply the combinatorial functions, and rebuild a correct mesh. This is handled by the `SurfaceSubdivision` and the `VolumeSubdivision` classes, which are designed to be used in a `with` block:

```python
import mouette as M

surface = M.mesh.load("/path/to/my/mesh.obj")

# At the start of the with block, the mesh is transformed into a RawMeshData and its connectivity is disabled
with M.mesh.SurfaceSubdivision(surface) as editor:
    # Call here the subdivision functions
    editor.subdivide_triangles_3quads()
    editor.triangulate()
# When exiting the 'with' block, the mesh is transformed back into a SurfaceMesh
# and its connectivity is regenerated
M.mesh.save(surface, "path/to/my/new/mesh.obj")
```

::: mouette.mesh.subdivision
    options:
        members:
            - split_edge
            - SurfaceSubdivision
            - split_double_boundary_edges_triangles
            - VolumeSubdivision
