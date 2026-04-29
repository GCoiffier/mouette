---
title: "Volume Frame Fields"
---

## VolumeFrameFields

To represent the frames in a 3D frame field, we use the [L4 Spherical Harmonics](mouette.geometry.SphericalHarmonics).

#### Usage
```python
from mouette import framefield
ff = framefield.VolumeFrameField(mesh, "vertices", 
      features=True, n_smooth=10., 
      smooth_attach_weight=0.1, 
      verbose=True)
ff.run()
ff.flag_singularities()
```

:::mouette.processing.framefield.VolumeFrameField


<!-- ## Saving from a file

save as a `.frame` file:

```
FRAME
nframes
a1x a1y a1z b1x b1y b1z c1x c1y c1z
a2x a2y a2z b2x b2y b2z c2x c2y c2z
...
anx any anz bnx bny bnz cnx cny cnz
```

## Reading from a file

```python
import mouette as M
from mouette.processing import VolumeFrameField

m = M.mesh.load("path/to/mesh")
ff = VolumeFrameField(m, "cells", verbose=True)
ff.read_from_file("path/to/frame")
ff.flag_singularities() 
ff_mesh = ff.export_as_mesh()
M.mesh.save(ff_mesh, "ff.mesh")
M.mesh.save(ff.singularity_graph, "singularity_graph.mesh")
``` -->