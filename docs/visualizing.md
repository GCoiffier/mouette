---
title: "Visualize geometry with third party softwares"
---

## Graphite 3

<!-- <figure markdown>
  ![](https://github.com/BrunoLevy/GraphiteThree/wiki/graphite_banner.gif){ width="800" }
</figure> -->


[https://github.com/BrunoLevy/GraphiteThree](https://github.com/BrunoLevy/GraphiteThree)



## Polyscope

<!-- <figure markdown>
  ![](https://polyscope.run/py//media/teaser.svg){ width="800" }
</figure> -->

[https://polyscope.run/py](https://polyscope.run/py)


```python
import polyscope as ps
import mouette as M
import numpy as np
bunny = M.mesh.load("bunny.obj")

ps.init()
V = np.asarray(bunny.vertices)
F = np.asarray(bunny.faces)
ps.register_surface_mesh("bunny", V, F)
ps.show()
```