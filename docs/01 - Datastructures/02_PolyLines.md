---
title: "Polylines"
---

Polylines are made of vertices (points in 3D space) and edges linking them. They allow to embed a graph in $\mathbb{R}^3$ and visualize it.

Edges are stored in a specific data container called `edges` as *ordered* pairs of vertex indices. They are not oriented and the smallest vertex index is always first. For example, $(1,3)$ is a valid edge index but $(3,1)$ is not. Edge orientation is performed automatically when instancing a `Polyline` from a `RawMeshData` (see [Edition and Manual Creation](../02%20-%20Manipulating%20Meshes/05_editing.md))

<figure markdown>
  ![Polyline Example](../_img/duck_polyline.jpeg){ width="400" }
  <figcaption>Example of a polyline</figcaption>
</figure>


## Polyline

::: mouette.mesh.datatypes.linear.PolyLine
    options:
      heading_level: 3

## Polyline Connectivity

::: mouette.mesh.datatypes.linear.PolyLine._Connectivity
    options:
      heading_level: 3