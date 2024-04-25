---
title: "Overview"
---

## Creating or loading geometry

mouette supports four types of geometries: point clouds (dim 0), polylines (dim 1), surface meshes (dim 2) and volume meshes (dim 3).


#### From numpy arrays

```python
import numpy as np
import mouette as M

vertices = np.array([
    [0,0,0],
    [1,2,3],
    [1,-1,0],
    [0,1,-2]
])
faces = np.array([[0,1,2], [0,1,3]])
surface = M.mesh.from_arrays(vertices,F=faces)
```

#### From raw mesh data

*/!\ Do not work directly append elements to containers of a SurfaceMesh, as this can create connectivity issues /!\\*

```python
data = M.mesh.RawMeshData()
data.vertices += [[0,0,0],[1,2,3],[1,-1,0],[0,1,-2]] # use += to append several elements
data.faces.append((0,1,2))
data.faces.append((0,1,3))
surface = M.mesh.SurfaceMesh(data) # create the object. This sanitizes the data under the hood and generates edges and face corners
```

#### Procedural Generation

```python
torus = M.procedural.torus(50,20)
sphere = M.procedural.sphere_uv(40,30)
```

#### From a file

```python
import mouette as M

mesh = M.mesh.load("path/to/my/mesh/mesh.obj")
```

Supported file formats are:
- wavefront (.obj)
- medit (.mesh)
- ply
- off
- stl
- geogram (.geogram_ascii)
- xyz (for point clouds only)

## Saving a mesh

```python
M.mesh.save(my_mesh,"path/to/export/mesh.obj")
```

## Mesh connectivity


## Manipulate attributes

It is possible to define any quantity on mesh elements

```python
my_v_attribute = mesh.vertices.create_attribute("my_attribute", float) # an attribute storing one floating-point number per vertex
my_v_attribute[3] = 4.

my_f_attribute = mesh.faces.create_attribute("my_attribute", 2, int) # an attribute storing two integers per face
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

```python
ff = M.processing.framefield.FrameField2DVertices(mesh) # define a frame field on the vertices of the surface mesh
ff.run()
ffmesh = ff.export_as_mesh()
M.mesh.save(ffmesh, "framefield.mesh")
```

Definition of classical discrete operators on meshes, like the gradient or the cotan-Laplacian:

```python
G = M.operators.gradient(mesh)
my_fun = mesh.vertices.get_attribute("f").as_array(len(mesh.vertices))
grad = G @ my_fun

L = M.operators.laplacian(mesh, cotan=True)
Lf = L @ my_fun
```