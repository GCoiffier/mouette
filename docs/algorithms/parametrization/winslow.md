---
title: "Foldover-free maps"
---

Implementation of the [_Foldover-free maps in 50 lines of code_](https://dl.acm.org/doi/10.1145/3450626.3459847) paper by Garanzha et al.

#### Usage

```python
import mouette as M
untangler = M.parametrization.WinslowInjectiveEmbedding(mesh, uv_init, lmbd=1.)
untangler.run()
```

See [this script](https://github.com/GCoiffier/mouette/blob/main/examples/winslow_untangle.py) for a full example.

#### Method

Given a triangulation $M=(V,T)$ of a disk-topology object and some initial $uv$-coordinates on the vertices of $M$, this method optimizes the $uv$-coordinates under fixed boundary so that no triangle is inverted in the final $uv$-mapping. This is done through the optimization of an energy function that acts on jacobian matrices $J \in \mathbb{R}^2$ of each triangle elements:

$$ \min_J f_\varepsilon(J) + g_\varepsilon(J)$$

where:

$$f_\varepsilon(J) = \frac{\text{tr}(J^TJ)}{\chi(\det J, \varepsilon)} \quad \quad 
g_\varepsilon(J) = \frac{\det{J}^2 + 1}{\chi(\det J,\varepsilon)}$$

and $\chi$ is a regularization function:

$$\chi(D, \varepsilon) = \frac{D + \sqrt{\varepsilon^2 + D^2}}{2}.$$

$\varepsilon$ is chosen during optimization as a decreasing sequence.

:::mouette.processing.parametrization.winslow
    options:
        heading_level: 2
        filters:
            - "!PointCloud"
            - "!PolyLine"
            - "!SurfaceMesh"
            - "!VolumeMesh"
            - "!check_argument"