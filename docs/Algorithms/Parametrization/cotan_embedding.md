---
title: "Cotan Embedding"
---

Usage:
```python
import mouette as M
from mouette.processing import parametrization

mesh = M.mesh.load("path/to/mesh")
cot_emb = parametrization.CotanEmbedding(mesh, [options])()
```

or, alternatively:
```python
cot_emb = parametrization.CotanEmbedding(mesh, [options])
cot_emb.run()
```

:::mouette.processing.parametrization.cotan_emb.CotanEmbedding