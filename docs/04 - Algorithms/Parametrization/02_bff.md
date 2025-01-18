---
title: "Boundary First Flattening"
---


#### Example
```python
import mouette as M

mesh = M.mesh.load("path/to/mesh")
bff = M.parametrization.BoundaryFirstFlattening(mesh, bnd_scale_fctr=scale, verbose=True)
mesh = bff.run() # /!\ mesh is not modified in place
```

See [https://github.com/GCoiffier/mouette/blob/main/examples/parametrization/bff.py](https://github.com/GCoiffier/mouette/blob/main/examples/parametrization/bff.py)

:::mouette.processing.parametrization.bff
    options:
        heading_level: 2
        members:
        - BoundaryFirstFlattening