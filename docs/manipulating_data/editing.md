---
title: "Edition and Manual Creation"
---

Alongside loading and saving various file formats, `mouette` allows to create datastructures from scratch. This can be done using the `RawMeshData` class. This class possesses all possible data containers available in mouette but no connectivity or functionnality: it is purely a storage class. The user can append its data to the corresponding containers:

```python
data = M.mesh.RawMeshData()
# use += instead of append to add several elements
data.vertices += [[0,0,0],[1,2,3],[1,-1,0],[0,1,-2]] 
data.edges.append((0,2))
data.faces.append((0,1,2))
data.faces.append((0,1,3))
```

When all operations are done, the `RawMeshData` object can be passed as an argument to the constructor of a [`PointCloud`][mouette.mesh.datatypes.pointcloud.PointCloud], [`Polyline`](../../01 - Datastructures/02_PolyLines/#polyline), [`SurfaceMesh`](../../01 - Datastructures/03_SurfaceMeshes/#surfacemesh) or [`VolumeMesh`](../../01 - Datastructures/04_VolumeMeshes/#volumemesh) object. This will sanitize the data under the hood and generate eventual corner data:

```python
# Create the SurfaceMesh object. 
# This sanitizes the data under the hood and generates edges and face corners
surface = M.mesh.SurfaceMesh(data) 
```

!!! Warning
    Do not directly append elements to containers of a `SurfaceMesh` or `VolumeMesh`, as this can create connectivity issues. `PointCloud` and `Polyline` are usually safe but using `RawMeshData` in all cases is the recommanded approach.
    
    For example, this code will create connectivity issues as the corresponding face corners will not be generated:
    ```python
    surface = M.mesh.SurfaceMesh()
    surface.vertices += [[0,0,0],[1,2,3],[1,-1,0],[0,1,-2]]
    surface.faces.append((0,1,2))
    surface.faces.append((0,1,3)) # This does not generate face corners correctly 
    ```

    While this is correct
    ```python
    raw_surface = M.mesh.RawMeshData()
    raw_surface.vertices += [[0,0,0],[1,2,3],[1,-1,0],[0,1,-2]]
    raw_surface.faces.append((0,1,2))
    raw_surface.faces.append((0,1,3)) 
    surface = M.mesh.SurfaceMesh(raw_surface) # This will create everything
    ```

## RawMeshData

::: mouette.mesh.mesh_data.RawMeshData
    options:
      heading_level: 3