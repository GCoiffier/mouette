---
title: "Frame Fields"
---

In its most general form, a frame field is a set of directions that live in the tangent space of a manifold.


Mouette implements frame fields via the abstract base class `FrameField`. This base class is instanciated with various algorithms using higher level functions.

#### Framefield optimization

Given a `FrameField` object initialized by some external function, the user can run the smoothing algorithm using the `.run()` method. Alternatively, a `FrameField` object is callable :

```python
ff()
ff.run() # produces similar results
```

#### Variables access

`FrameField` objects are iterable. Indexing a frame field returns the representation of the frame at the given index. For 2D frame fields, this representation is a complex number. For 3D frame fields, it is a numpy array of shape (9,) representing the corresponding [L4 spherical harmonics](../../Geometry/spherical_harmonics.md).

```python
ff[7] # returns the representation of the frame for element (vertex/face/cell )
```

#### Singularity flagging

The `.flag_singularities()` method of the `FrameField` class allow to detect any singular point inside the frame field. When the boundary of the domain is constrained, those singularity points are bound to appear in 2D (due to the [Poincaré-Hopf theorem](https://en.wikipedia.org/wiki/Poincar%C3%A9%E2%80%93Hopf_theorem)). In 3D, the singularities are no longer ponctual, but form a network of lines called a singularity graph. See [the specific 3D case](03_3dFF.md) for more details.

#### Export and Visualization

The `export_as_mesh()` method of the `FrameField` class outputs frames either as little crossed in 2D or little cubes in 3D for visualization purposes.