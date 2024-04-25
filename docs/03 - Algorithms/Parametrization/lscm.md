---
title: "Least-Square Conformal Maps"
---

Usage:
```python
import mouette as M
from mouette.processing import parametrization

mesh = M.mesh.load("path/to/mesh")
lscm = parametrization.LSCM(mesh, [options])()
```

or, alternatively:
```python
lscm = parametrization.LSCM(mesh, [options])
lscm.run()
```

:::mouette.processing.parametrization.lscm.LSCM