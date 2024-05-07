---
title: "Overview"
---

## Creating or loading geometry

mouette supports four types of geometries: point clouds (dim 0), polylines (dim 1), surface meshes (dim 2) and volume meshes (dim 3).


```python
import numpy as np
import mouette as M

### From numpy arrays
vertices = np.array([[0,0,0],[1,2,3],[1,-1,0],[0,1,-2]])
faces = np.array([[0,1,2], [0,1,3]])
surface1 = M.mesh.from_arrays(vertices,F=faces) # a simple quad

### From files
surface2 = M.mesh.load("path/to/my/mesh/mesh.obj")

### From provided procedural functions
surface3 = M.procedural.torus(50,20)
```

`mouette` supports a variety of file formats: wavefront (.obj), medit (.mesh), stl, ply, off, tet, geogram (.geogram_ascii) and xyz (for point clouds).

## Saving a mesh

Data structures from `mouette` can be easily stored in various file formats using the `save` function:

```python
M.mesh.save(my_mesh,"path/to/export/mesh.obj")
```

the extension given in the path determines the file format.

## Mesh connectivity

Starting from the `Polyline` class, `mouette` implements various methods to query the adjacent elements of a given vertex, edge, face or cell:

```python

# list of vertices that are linked by an edge to vertex 42
neighbors = my_mesh.connectivity.vertex_to_vertices(42) 

# list of faces in which vertex 42 is a vertex
face_ring = my_mesh.connectivity.vertex_to_faces(42) 
```

## Manipulating attributes

It is possible to define any quantity on mesh elements:

```python
# an attribute storing one floating-point number per vertex
my_v_attribute = mesh.vertices.create_attribute("my_attribute", float) 
my_v_attribute[3] = 4.

# an attribute storing two integers per face
my_f_attribute = mesh.faces.create_attribute("my_attribute", 2, int) 
m_f_attribute[2] = [1,3]
```

Attributes can store booleans, integers, floating-point numbers, complex numbers and strings, using the provided python types `bool`, `int`, `float`, `complex` and `str`.

Mouette also implements classical quantities to be computed on a mesh as attributes in the `M.attributes`submodule:

```python
degree = M.attributes.vertex_degree(mesh) # number of neighbors of each vertex
lengths = M.attributes.edge_length(mesh) # length of each edge
areas = M.attributes.face_area(mesh) # area of each face
angles = M.attributes.corner_angles(mesh) # angle at each face corner
```


## Apply Geometry Processing Algorithms

`mouette` implements some geometry processing algorithms to be applied to your data:

```python
# define a frame field on the vertices of the surface mesh
ff = M.processing.framefield.FrameField2DVertices(mesh) 
ff.run()
ffmesh = ff.export_as_mesh()
M.mesh.save(ffmesh, "framefield.mesh")
```

Classical discrete operators on meshes, like the gradient or the cotan-Laplacian, are also defined:

```python
G = M.operators.gradient(mesh)
my_fun = mesh.vertices.get_attribute("f").as_array()
grad = G @ my_fun

L = M.operators.laplacian(mesh, cotan=True)
Lf = L @ my_fun
```