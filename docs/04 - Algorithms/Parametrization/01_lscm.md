---
title: "Least-Square Conformal Maps"
---


#### Example
```python
import mouette as M

mesh = M.mesh.load("path/to/mesh")
lscm = M.parametrization.LSCM(mesh, [options])
lscm.run()
```
See [https://github.com/GCoiffier/mouette/blob/main/examples/tutte.py](https://github.com/GCoiffier/mouette/blob/main/examples/lscm.py)

:::mouette.processing.parametrization.lscm