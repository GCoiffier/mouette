---
title: "Dependencies"
---

#### Numpy

Mouette makes use of numpy and scipy for the representation of dense

#### Scipy

Representation of sparse matrices

#### Numba

Numba is used for speeding up repetitive computations like loops over elements in some algorithms.

#### OSQP

OSQP ( Operator Splitting Quadratic Program ) is a quadratic programming algorithm supporting linear equalities and inequalities as constraints. Mouette relies on this solver for least-square problems when linear constraints are involved.

See the website [https://osqp.org/](https://osqp.org/) for a complete documentation

#### IO libraries

- stl-reader 

- pyminiply