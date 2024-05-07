---
title: "Tutte's Embedding"
---

#### Example
```python
import mouette as M
mesh = M.mesh.load(args.model)
tutte = M.parametrization.TutteEmbedding(mesh, [boundary_mode], use_cotan=True, verbose=True)
tutte.run()
M.mesh.save(mesh, "tutte_model.obj")
M.mesh.save(tutte.flat_mesh, "tutte_flat.obj")
```

See [https://github.com/GCoiffier/mouette/blob/main/examples/tutte.py](https://github.com/GCoiffier/mouette/blob/main/examples/tutte.py)

:::mouette.processing.parametrization.tutte