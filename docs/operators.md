---
title: "Matrix operators"
weight: 2
---

## Connectivity operators

:::mouette.operators.adjacency
    options:
        heading_level: 2
        filters:
            - "!PointCloud"
            - "!PolyLine"
            - "!SurfaceMesh"
            - "!VolumeMesh"
            - "!check_argument"

## Gradient operator

```python
    G = M.operators.gradient(mesh)
    my_fun = mesh.vertices.get_attribute("f").as_array()
    grad = G @ my_fun
```

:::mouette.operators.gradient
    options:
        heading_level: 3
        members:
            - gradient


## Laplacian operator
[https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Mesh_Laplacians](https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Mesh_Laplacians)

We refer to [this course](https://www.cs.cmu.edu/~kmcrane/Projects/Other/SwissArmyLaplacian.pdf) for a great overview of the Laplacian operator and its use in geometry processing.

For the generalization to volumes, see [this pdf](https://www.cs.cmu.edu/~kmcrane/Projects/Other/nDCotanFormula.pdf)

#### Example

```python
import mouette as M
import numpy as np

mesh = M.mesh.load("path/to/my/mesh/mesh.obj")
W = M.operators.laplacian(mesh,cotan=True)
A = M.operators.area_weight_matrix(mesh, inverse=True)
L = A @ W # discretization of the laplace-beltrami operator is inverted mass matrix times the cotan weight matrix
X = np.random.random(len(mesh.vertices)) # a value in (0,1) per vertex
for _ in range(10):
    X = W.dot(X) # perform 10 steps of diffusion
```

:::mouette.operators.laplacian_op
    options:
        heading_level: 3
        filters:
            - "!PointCloud"
            - "!PolyLine"
            - "!SurfaceMesh"
            - "!VolumeMesh"
            - "!check_argument"