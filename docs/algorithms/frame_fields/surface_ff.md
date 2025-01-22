---
title: "Surface Frame Fields"
---

## SurfaceFrameField

#### Usage
```python
from mouette import framefield
ff = framefield.SurfaceFrameField(mesh, "faces", 
      order=4, features=True, 
      verbose=True, n_smooth=10, 
      smooth_attach_weight=0.2, 
      cad_correction=True)
```

::: mouette.processing.framefield.SurfaceFrameField