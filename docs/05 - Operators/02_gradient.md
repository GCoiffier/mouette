---
title: "Gradient"
---

#### Example

```python
G = M.operators.gradient(mesh)
my_fun = mesh.vertices.get_attribute("f").as_array()
grad = G @ my_fun
```

See [https://github.com/GCoiffier/mouette/blob/main/examples/gradient.py](https://github.com/GCoiffier/mouette/blob/main/examples/gradient.py)

:::mouette.operators.gradient