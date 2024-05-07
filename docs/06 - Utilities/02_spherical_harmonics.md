---
title: "L4 Spherical Harmonics"
---

[Spherical Harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics) are a basis of functions over which any function defined on the sphere can be decomposed. In mouette, we make use of the nine frequency L=4 harmonics to [represent orthogonal frame fields in 3D](../04%20-%20Algorithms/Frame%20Fields/03_3dFF.md). The `geometry.SphericalHarmonics` utility allow to manipulate such objects.

<figure markdown>
  ![Representation of first spherical harmonics](https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Spherical_Harmonics.png/1280px-Spherical_Harmonics.png){ width="500" }
  <figcaption> Spherical harmonics from frequency L=1 to L=4 (from Wikipedia) </figcaption>
</figure>

:::mouette.geometry.SphericalHarmonics