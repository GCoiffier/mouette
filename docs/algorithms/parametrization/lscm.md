---
title: "Least-Square Conformal Maps"
---

[https://en.wikipedia.org/wiki/Least_squares_conformal_map](https://en.wikipedia.org/wiki/Least_squares_conformal_map)

#### Usage
```python
import mouette as M

mesh = M.mesh.load("path/to/mesh")
lscm = M.parametrization.LSCM(mesh, [options])
lscm.run()
```

:::mouette.processing.parametrization.lscm
    options:
        heading_level: 2
        filters:
        - "!PointCloud"
        - "!PolyLine"
        - "!SurfaceMesh"
        - "!VolumeMesh"
        - "!check_argument"