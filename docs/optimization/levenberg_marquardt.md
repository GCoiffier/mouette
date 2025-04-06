---
title: "Levenberg-Marquardt"
---

Given a function $F : \mathbb{R}^n \rightarrow \mathbb{R}^m$ of class $\mathcal{C}^1$, the Levenberg-Marquardt optimizer iteratively solves:

$$ \min_{x \in \mathbb{R}^n} \frac{1}{2} ||F(x)||^2$$

## Principle
Let $x_k$ be the variable vector at iteration $k$. 
Define $m_\lambda^k : \mathbb{R}^n \rightarrow \mathbb{R}$ by 

$$\begin{align}
m_\lambda^k(x) &= \frac{1}{2}|| F_k + J_k (x - x_k) ||^2 + \frac{\lambda}{2} || x - x_k||^2 \\
&= \frac{1}{2}|| F_k + (J_k + \lambda I) (x - x_k) ||^2
\end{align}
$$

where $F_k = F(x_k)$ is the value of the function at point $x_k$ and $J_k = J(x_k)$ is the value of its Jacobian matrix at point $x_k$.

Set $x_{k+1} = \text{argmin}_x m_\lambda^k(x)$ for some $\lambda = \lambda_k \in \mathbb{R}^+$

And iterate until convergence.


## Example

```python

import mouette as M
import numpy as np
import scipy.sparse as sp

def function_to_optimize(x, a, b):
    # Rosenbrock function and gradient
    f = function(x,a,b)
    J = sp.csr_matrix([2*(x[0]-a) - 4*b*x[0]*(x[1]-x[0]*x[0]),  2*b*(x[1]-x[0]*x[0])]).reshape((1,2))
    return f,J

optimizer = M.optimize.LevenbergMarquardt()

# adjust hyperparameters from the HP attribute
optimizer.HP.ENERGY_MIN = 1E-10
optimizer.HP.MIN_GRAD_NORM = 0. 

# Register the function to optimize
optimizer.register_function(lambda x : function_to_optimize(x, 1., 10.), name="Rosenbrock")
optimizer.run(x_init=np.random.random((2,))) # run the optimization starting from a random position

print("Local minimum found:" , optimizer.X)
print("Expected minimum :", [A,A])
```

## Implementation

:::mouette.optimize.levenberg_marquardt
    options:
        heading_level: 3
        filters:
        - "!PolyLine"
        - "!SurfaceMesh"
        - "!VolumeMesh"
        - "!check_argument"