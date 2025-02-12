---
title: "Volume meshes"
---

<figure markdown>
  ![Volume Mesh Example](../_img/sculpt_tets.jpeg){ width="600" }
  <figcaption>A tetrahedral mesh</figcaption>
</figure>

## VolumeMesh

::: mouette.mesh.datatypes.volume.VolumeMesh
    options:
      heading_level: 3

## Volume Connectivity

!!! Warning

    Volume connectivity is a work in progress and will be available soon.

<!-- ::: mouette.mesh.datatypes.volume.VolumeMesh._Connectivity
    options:
      heading_level: 3 -->


## BoundaryConnectivity

In some applications, you may need to access the boundary surface mesh of a volume mesh in a self-contained way while keeping links to the original volume. It allows requests like "return the list of all vertices adjacent to boundary vertex i that are on the boundary"

It implements the same methods as the [Connectivity class of surface meshes](./SurfaceMeshes.md#surface-connectivity) for the boundary mesh, as well as indirection arrays to move from volume indices to surface indices. Returned indices of connectivity queries are indices with relation to the volume mesh.

To save time and memory, the boundary indirections are not computed by default. Use the `enable_boundary_connectivity()` method of the `VolumeMesh` to explicitly generate a `BoundaryConnectivity` object, stored in the `.boundary_connectivity` attribute.

!!! Example
    ```python
    import mouette as M
    m = M.mesh.load("my_volume_mesh.tet")

    # if enable_boundary_connectivity is not called, `m.boundary_connectivity` is None
    m.enable_boundary_connectivity() # builds the indirection
    bnd_m = m.boundary_mesh # access the boundary mesh as an independent surface mesh
    boundary_neighbors = m.boundary_connectivity.vertex_to_vertices(4) # query connectivity
    ```

::: mouette.mesh.datatypes.volume.VolumeMesh._BoundaryConnectivity
    options:
      heading_level: 3
      members:
        - clear
