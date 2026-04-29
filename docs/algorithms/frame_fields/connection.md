---
title: "Discrete Connections"
---

A connection is an object from differential geometry that maps between tangent planes. On a manifold, tangent planes are the local planar approximation of the surface. When comparing objects living in two adjacent planes, one must first align the plane and their corresponding bases. This can be done by _parallel transporting_ objects from one plane to the other.

On a surface mesh, this process is discrete and not differential. The parallel transport is then simply the angle formed by the two bases of the two planes.

Interesting blog post on the topic: [http://wordpress.discretization.de/geometryprocessingandapplicationsws19/connections-and-parallel-transport/](http://wordpress.discretization.de/geometryprocessingandapplicationsws19/connections-and-parallel-transport/)


## SurfaceConnectionVertices

:::mouette.processing.connection.SurfaceConnectionVertices
    options:
        heading_level: 3

## SurfaceConnectionFaces

:::mouette.processing.connection.SurfaceConnectionFaces
    options:
        heading_level: 3

## DiscreteExponentialMap
[https://en.wikipedia.org/wiki/Exponential_map_(Riemannian_geometry)](https://en.wikipedia.org/wiki/Exponential_map_(Riemannian_geometry))

#### Usage
```python
conn = SurfaceConnectionVertices(mesh)
expm = DiscreteExponentialMap(mesh, conn, rad)
expm.run({0, 2, 42}) # computes map for vertices 0, 2 and 42

expm.run() # computes map for all vertices

u,v = expm.map(0, 3) # coordinates of vertex 3 in exp map of vertex 0
u,v = expm.map(3, 0) # coordinates of vertex 0 in exp map of vertex 3. Exp map of 3 is computed on the go if necessary
```

:::mouette.processing.expmap.DiscreteExponentialMap
    options:
        heading_level: 3