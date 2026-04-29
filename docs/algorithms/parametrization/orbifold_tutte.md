---
title: "Orbifold Tutte Embeddings"
---

Implementation of the [Orbifold Tutte Embedding](https://noamaig.github.io/html/projects/orbifold/orbifold_lowres.pdf) technique from Noam Aigerman and Yaron Lipman.


This method embeds a sphere-topology mesh into an Euclidean orbifold, that is a chart that paves the plane up to some rotation. In practice, it defines cuts between a set of prescribed cones on the surface which become virtual seams. The parametrization's distortion only depends on the position of the cones.

They are four ways to perform such an embedding:  
1) Square orbifold : 3 cones of $\pi/2$, $\pi$ and $\pi/2$  
2) Diamond orbifold : 3 cones $2\pi/3$, $2\pi/3$ and $2\pi/3$  
3) Triangle orbifold : 3 cones of $\pi$, $2\pi/3$ and $\pi/3$  
4) Parallelogram orbifold : 4 cones of $\pi$, $\pi$, $\pi$ and $\pi$  

<figure markdown>
  ![A visual description of the 4 types of Euclidean orbifolds and how their sides correspond to one another](../../_img/orbifolds.png){ width="600" }
  <figcaption>Four types of orbifolds (respectively square, diamond, triangle and parallelogram). Figure from the paper.</figcaption>
</figure>

!!! Note
    Only the "square" and the "parallelogram" are currently implemented in `mouette`.

#### Usage
```python
import mouette as M
mesh = M.mesh.load(args.model)
orbifold_type = "parallelogram" # 4 cones, or "square" for 3 cones
orbTutte = M.parametrization.OrbifoldTutteEmbedding(mesh, orbifold_type, cones)
orbTutte.run()
```

:::mouette.processing.parametrization.orbifold_tutte
    options:
        heading_level: 2
        filters:
        - "!PointCloud"
        - "!PolyLine"
        - "!SurfaceMesh"
        - "!VolumeMesh"
        - "!check_argument"
        - "!BoundaryMode"