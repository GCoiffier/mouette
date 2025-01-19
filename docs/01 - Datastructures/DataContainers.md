---
title: "Data Containers"
weight: 1
---

`mouette` is architectured around the concept of _data containers_, which are containers dedicated to the storage of one type of elements in a mesh and all of its attributes. Data structures in mouette are then defined as collections of specific data containers:

- A [`PointCloud`][mouette.mesh.datatypes.pointcloud.PointCloud] object only has one container named `vertices` such that `object.vertices[i]` is a 3D vector containing the position of vertex `i` in space.

- In addition to the `vertices` container, a [`Polyline`](../02_PolyLines/#polyline) object also defines an `edges` container, which stores pairs of vertex indices.

- A [`SurfaceMesh`](../03_SurfaceMeshes/#surfacemesh) object adds the `faces` and `face_corners` containers.

- Finally, a [`VolumeMesh`](../04_VolumeMeshes/#volumemesh) objects adds a `cells` container and two corner containers: `cell_corners` and `cell_faces`.

## The `DataContainer` class

::: mouette.mesh.data_container.DataContainer
    options:
      heading_level: 3

::: mouette.mesh.data_container._BaseDataContainer
    options:
      heading_level: 3
      
## The special case of corner containers

::: mouette.mesh.data_container.CornerDataContainer
    options:
      heading_level: 3

## Attributes

::: mouette.mesh.mesh_attributes
    options:
      heading_level: 3