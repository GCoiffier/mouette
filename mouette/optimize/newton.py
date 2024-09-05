import numpy as np
from osqp import OSQP
from dataclasses import dataclass
import scipy.sparse as sp
from scipy.optimize import line_search

from tqdm import tqdm, trange
from tqdm.utils import _term_move_up
prefix = _term_move_up() + '\r'

from ..utils import Logger, get_osqp_lin_solver
from .. import geometry as geom

@dataclass
class NewtonParameters:
    """
    Hyper parameters for Newton's algorithm
    """
    N_ITER_MAX    : int   = 1000 # stopping criterion on maximal number of iterations
    MIN_STEP_NORM : float = 1e-6 # stopping criterion, on ||x_n - x_{n-1}||
    MIN_DELTA_F   : float = 1e-8 # stopping criterion on ||f_n - f_{n-1}||
    MIN_GRAD_NORM : float = 1e-8 # stopping criterion on gradient projected on constraints

    alpha_init : float = 1. # starting step size in linesearch
    c1 : float = 1e-4 # Weak Wolfe 1st condition parameter
    rho : float = 0.5 # shrinkage parameter in linesearch

@dataclass
class NewtonVerboseOptions:
    logger_verbose : bool = True  # main verbose for mouette logging
    solver_verbose : bool = False # prints outputs of OSQP solver
    use_tqdm       : bool = False  # progression bar using tqdm
    log_frequency  : int  = 1     # display an info message every 'log_frequency' iterations

class Newton(Logger):
    """
    Newton's algorithm to find the minimum of a function knowing its gradient and hessian.

    References:
        [1] https://won-j.github.io/M1399_000200-2021fall/lectures/22-newton/newton_constr.html
    """

    def __init__(self,
        HP : NewtonParameters = NewtonParameters(), 
        verbose : NewtonVerboseOptions = NewtonVerboseOptions(),
        **kwargs):
        super().__init__("Newton",verbose.logger_verbose)
        self.verbose_options = verbose
        self.HP = HP

        self.X = None # variables

        ### Linear Constraints
        self._cstM = None
        self._cstL = None
        self._cstU = None

        ### Function to optimize
        self._func = None
        self._func_no_grad = None

        ### Others
        self._stop_criterion_instance = None # OSQP instance to compute projected gradient norm

        ### Additionnal parameters
        self._linsys_solver = kwargs.get("lin_solver", get_osqp_lin_solver())

    def register_constraints(self, A : sp.spmatrix, l : np.ndarray = None, u : np.ndarray = None):
        """
        Adds linear constraints to the optimization problem :

        min_X  F(X) s.t.  l <= AX <= u

        Args:
            cstMat (sp.spmatrix): constraint matrix, either dense or sparse
            cstL (np.ndarray, optional): vector l. Defaults to None.
            cstR (np.ndarray, optional): vector u. Defaults to None.
        """

        if isinstance(A, sp.spmatrix):
            self._cstM = A.tocsc()
        elif isinstance(A, np.ndarray):
            assert len(A.shape)==2 # we make sure A is indeed a matrix
            self._cstM = sp.csc_matrix(A)
        self._cstL = l
        self._cstU = u

    def register_function(self, fun, fun_noG=None):
        """
        Adds a function to minimize.

        Args:
            fun (python callable): A function taking a single argument X (np.array) returning a real value F, a gradient G and a hessian matrix H

            fun_noG (python callable, optional): The same function that avoids the computation of the gradient and the hessian for speed purposes. If not provided, will use the function provided above and ignore the other arguments.
        """
        self._func = fun
        if fun_noG is None:
            fun_noG = lambda x : fun(x)[0]
        self._func_no_grad = fun_noG

    def energy(self, X=None, derivatives=False):
        if X is None: X = self.X
        if derivatives:
            return self._func_no_grad(X)
        return self._func(X)

    def optimize(self, x_init: np.ndarray) -> float:
        """
        Alias of the 'run' method.

        Args:
            x_init (np.ndarray): initial values for the variables.

        Returns:
            float: final value of the energy
        """
        return self.run(x_init)

    def run(self, x_init : np.ndarray):
        """
        Runs the optimizer

        Args:
            x_init (np.ndarray): initial values for the variables.

        Returns:
            float: final value of the energy
        """

        self.X = self._find_closest_feasible_solution(x_init) # variable vector
        
        if self.HP.N_ITER_MAX <= 0 :
            # return current energy
            return self.energy(x_init, derivatives=False)

        if self._func is None: 
            # No functions to optimize
            self.log("No function to optimize. Register a function before calling the optimization.")
            return 0.

        # iterable for optimization steps
        if self.verbose_options.logger_verbose and self.verbose_options.use_tqdm:
            print()
            iterobj = trange(self.HP.N_ITER_MAX, total=self.HP.N_ITER_MAX, position=1, leave=False, ncols=100, unit="")
        else:
            iterobj = range(self.HP.N_ITER_MAX)

        try:
            # Iteration loop
            for it in iterobj:
                F,G,H = self.energy()                
                if np.isnan(F):
                    return self._end_optimization(F, "NaN value in energy")
                grad_stop, grad_norm = self._stop_criterion_projgrad(G)
                if grad_stop:
                    return self._end_optimization(F, "projected grad norm on constraints < min_grad_norm")

                osqp_instance = OSQP()
                if self._cstM is not None:
                    # OSQP solve inside iteration loop computes the increment x_{n-1} - x_n, so constraints have to be shifted
                    xCst = self._cstM @ self.X
                    cstL = self._cstL - xCst
                    cstU = self._cstU - xCst
                else:
                    cstL,cstU = None, None

                ### Compute direction of descent
                osqp_instance.setup(H, q=G, A=self._cstM, l=cstL, u=cstU,
                    verbose=self.verbose_options.solver_verbose,
                    eps_abs=1e-3, eps_rel=1e-3,
                    max_iter=100, polish=True, check_termination=10, 
                    adaptive_rho=True, linsys_solver= self._linsys_solver)
                s = osqp_instance.solve().x # computed increment
                if s[0] is None: print(s)

                ### Perform Linesearch
                # ln = line_search(self._func_no_grad, lambda x : self._func(x)[1], self.X, -G)
                # alpha, new_F = ln[0], ln[3]
                # success = (alpha is not None)
                
                # alpha,new_F,success = self._backtracking_linesearch(F,G,s)
                
                alpha, new_F, success = 1., self._func_no_grad(self.X+s), True

                if not success:
                    return self._end_optimization(F, "Linesearch failed")
                self.X += s*alpha
                RelDeltaF = abs(new_F-F)/abs(F)
                if RelDeltaF < self.HP.MIN_DELTA_F:
                    return self._end_optimization(F, "Relative progression of F < ΔF_min")
                step_norm = geom.norm(s*alpha)
                if step_norm < self.HP.MIN_STEP_NORM :
                    return self._end_optimization(F, "Step norm < min_step_norm")

                if self.verbose_options.logger_verbose and self.verbose_options.log_frequency >0 and it%self.verbose_options.log_frequency==0:
                    if self.verbose_options.use_tqdm:
                        tqdm_log = prefix
                        tqdm_log += "F: {:.2E} | Grad: {:.2E} | ".format(F, grad_norm)
                        tqdm_log += "ΔE {:.2E} | Step {:.2E}".format(RelDeltaF, step_norm)
                        tqdm_log += " " * 10
                        tqdm.write(prefix)
                        tqdm.write(tqdm_log)
                    else:
                        log = f"{it+1}/{self.HP.N_ITER_MAX} | "
                        log += "F: {:.2E} | Grad: {:.2E} | ".format(F, grad_norm)
                        log += "ΔE {:.2E} | Step {:.2E}".format(RelDeltaF, step_norm)
                        print(log)

        except KeyboardInterrupt:
            self.log("Manual interruption")
        return self._end_optimization(F, "max iteration reached")

    def _find_closest_feasible_solution(self, x_init):
        if self._cstM is None: 
            # problem is unconstrained, solution is feasible
            return np.copy(x_init)
        nvar = x_init.size
        if self._cstM.shape[1] != nvar:
            raise Exception(f"Constraint matrix size {self._cstM.shape} and number of variables {nvar} mismatch.")
        instance = OSQP()
        instance.setup(sp.eye(nvar, format="csc"), q=-x_init, 
                    A=self._cstM, l=self._cstL, u=self._cstU, 
                    verbose=True, linsys_solver= self._linsys_solver)
        x_proj = instance.solve().x
        return x_proj

    def _backtracking_linesearch(self, F, G, s):
        a = self.HP.alpha_init
        f_step = self._func_no_grad(self.X+a*s)
        Gs = np.dot(G,s)
        n_iter = 0
        n_iter_max = 100
        while f_step >= F + self.HP.c1 * a * Gs and n_iter<n_iter_max:
            a = self.HP.rho * a
            f_step = self._func_no_grad(self.X + a*s)
            n_iter += 1
        return a, f_step, (n_iter < n_iter_max)

    def _stop_criterion_projgrad(self, G):
        # Projected gradient on the orthogonal of space spanned by constraints should have small norm
        if self._cstM is None:
            xnorm = np.sqrt(np.dot(G,G))
        else:
            n = self._cstM.shape[1]
            if self._stop_criterion_instance is None:
                self._stop_criterion_instance = OSQP()
                self._stop_criterion_instance.setup(
                    sp.eye(n,format=("csc")), q=-2*G, A = self._cstM, l = self._cstL, u = self._cstU,
                    verbose=self.verbose_options.solver_verbose)
            else:
                self._stop_criterion_instance.update(q=-2*G)
            x = self._stop_criterion_instance.solve().x
            if x is None: return False, 0.
            xnorm = np.sqrt(np.dot(x,x))
        return xnorm<self.HP.MIN_GRAD_NORM, xnorm

    def _end_optimization(self, energy, message):
        if self.verbose:
            if self.verbose_options.use_tqdm: print("\n")
            self.log("End of optimization: "+message)
            self.log(f"Final Energy: {energy:.4E}")
            if self.verbose_options.use_tqdm: print("\r")
        return energy        