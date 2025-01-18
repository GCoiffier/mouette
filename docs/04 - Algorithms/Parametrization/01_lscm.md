---
title: "Least-Square Conformal Maps"
---

[https://en.wikipedia.org/wiki/Least_squares_conformal_map](https://en.wikipedia.org/wiki/Least_squares_conformal_map)

#### Example
```python
import mouette as M

mesh = M.mesh.load("path/to/mesh")
lscm = M.parametrization.LSCM(mesh, [options])
lscm.run()
```
See [https://github.com/GCoiffier/mouette/blob/main/examples/parametrization/lscm.py](https://github.com/GCoiffier/mouette/blob/main/examples/parametrization/lscm.py)

:::mouette.processing.parametrization.lscm
    options:
        heading_level: 2
        filters:
        - "!PointCloud"
        - "!PolyLine"
        - "!SurfaceMesh"
        - "!VolumeMesh"
        - "!check_argument"