---
weight: -10
---

Mouette is a small python library for handling point clouds, polylines, surface and volume meshes. It also contains various geometry processing algorithm, like shortest-paths, old-school parametrization or frame field computations.

Mouette (French for seagull) stands for _Maillages, OUtils Et Traitement auTomatique de la géométriE_ (French for "Meshes, Tools and Geometry Processing).

## Installation

Using pip: 
```
pip install mouette
```

## Dependencies

#### Numpy and Scipy
Mouette makes use of numpy and scipy for the representation of dense and sparse matrices.

#### Numba

Numba is used for speeding up repetitive computations like loops over elements in some algorithms.

#### OSQP

OSQP ( Operator Splitting Quadratic Program ) is a quadratic programming algorithm supporting linear equalities and inequalities as constraints. Mouette relies on this solver for least-square problems when linear constraints are involved.

See the website https://osqp.org/ for a complete documentation

#### IO libraries
    stl-reader
    plyfile