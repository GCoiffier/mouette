Mouette is a small python library for handling point clouds, polylines, surface and volume meshes. It also contains various geometry processing algorithm, like shortest-paths, old-school parametrization or frame field computations.

Mouette (French for seagull) stands for _Maillages, OUtils Et Traitement auTomatique de la géométriE_ (French for "Meshes, Tools and Geometry Processing).

## Installation

Using pip: 
```pip install mouette```

## Overview

```python
import mouette as M

mesh = M.mesh.load("path/to/my/mesh/mesh.obj")
M.mesh.save(mesh,"path/to/export/mesh.obj")
```

### Run tests

`python -m pytest tests/`
