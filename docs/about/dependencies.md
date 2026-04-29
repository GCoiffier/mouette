---
title: "Dependencies"
---

#### Numpy and Scipy

Mouette makes use of [numpy](https://numpy.org/) arrays for storing data, as well as the [scipy's sparse module]()

#### Numba

[Numba](https://numba.pydata.org/) is a just-in-time compiler used for speeding up computations as well as making them parallel.

#### OSQP

[OSQP](https://osqp.org/)( Operator Splitting Quadratic Program ) is a quadratic programming algorithm supporting linear equalities and inequalities as constraints. Mouette relies on this solver for least-square problems when linear constraints are involved.

#### IO libraries

- [stl-reader](https://pypi.org/project/stl-reader/)
- [pyminiply](https://pypi.org/project/pyminiply/)


#### Misc

- [aenum](https://pypi.org/project/aenum/) (better enumerations)
- [tqdm](https://tqdm.github.io/) (cool progress bars)