---
title: "Boundary First Flattening"
weight: 3
---


#### Example
```python
import mouette as M

mesh = M.mesh.load("path/to/mesh")
bff = M.parametrization.BoundaryFirstFlattening(mesh, bnd_scale_fctr=scale, verbose=True)
mesh = bff.run() # /!\ mesh is not modified in place
```

:::mouette.processing.parametrization.bff
    options:
        heading_level: 2
        members:
        - BoundaryFirstFlattening