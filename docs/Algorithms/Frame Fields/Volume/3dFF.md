---
title: "3D Frame Fields"
---

# 3D Frame Fields

```python
from mouette.processing import framefield
```

In mouette, two algorithms of smoothest frame field in 3D have been implemented : `FrameField3DCells` and `FrameField3DVertices`


## Frame field on vertices

## Frame field on cells

```python
import mouette as M
from mouette.processing import framefield

m = M.mesh.load("path/to/mesh")
ff = framefield.FrameField3DCells(m, verbose=True)
ff.run() # Initializes values 
ff.flag_singularities() # Computes the singularity graph
```

Singularity graph can then be accessed as a polyline through `ff.singularity_graph`

## Saving from a file

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

m = M.mesh.load("path/to/mesh")
ff = M.processing.framefield.FrameField3DCells(m)
ff.read_from_file("path/to/frame")
ff.flag_singularities() 
ffm = ff.export_as_mesh()
M.mesh.save(ffm, "ff.mesh")
M.mesh.save(ff.singularity_graph, "singuls.mesh")
```
