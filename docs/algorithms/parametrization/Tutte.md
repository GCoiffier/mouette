---
title: "Tutte's Embedding"
---

#### Usage
```python
import mouette as M
mesh = M.mesh.load(args.model)
tutte = M.parametrization.TutteEmbedding(mesh, [boundary_mode], use_cotan=True, verbose=True)
tutte.run()
M.mesh.save(mesh, "tutte_model.obj")
M.mesh.save(tutte.flat_mesh, "tutte_flat.obj")
```

## TutteEmbedding

:::mouette.processing.parametrization.tutte.TutteEmbedding
    options:
        heading_level: 3
        filters:
        - "!PointCloud"
        - "!PolyLine"
        - "!SurfaceMesh"
        - "!VolumeMesh"
        - "!check_argument"
        - "!BoundaryMode"