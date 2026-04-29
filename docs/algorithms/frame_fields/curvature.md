---
title: "Directions of Curvature"
---

The principal directions of curvature are the two tangent eigenvectors of the curvature tensor of a surface 
([https://en.wikipedia.org/wiki/Principal_curvature](https://en.wikipedia.org/wiki/Principal_curvature)). In practice, they are always orthogonal to each other and point towards the directions where curvature varies the fastest and the slowest.

## PrincipalDirections

#### Usage
```python
from mouette import framefield
ff = framefield.PrincipalDirections(mesh, "vertices", features=True, verbose=True, n_smooth=3)
ff.run()
ff.flag_singularities()
```

:::mouette.processing.framefield.PrincipalDirections