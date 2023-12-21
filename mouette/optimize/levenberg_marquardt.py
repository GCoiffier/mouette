import numpy as np
from osqp import OSQP
from dataclasses import dataclass
import scipy.sparse as sp
from types import FunctionType

from tqdm import tqdm, trange
from tqdm.utils import _term_move_up
prefix = _term_move_up() + '\r'

from ..utils import Logger, get_osqp_lin_solver
from .. import geometry as geom

@dataclass
class LMParameters:
    """
    Hyper parameters for the Levenberg-Marquardt algorithm
    """
    N_ITER_MAX    : int   = 1000 # stopping criterion on maximal number of iterations
    ENERGY_MIN    : float = 1e-7 # stopping criterion for reaching zero
    MIN_STEP_NORM : float = 1e-5 # stopping criterio, on ||x_n - x_{n-1}||
    MIN_GRAD_NORM : float = 1e-5 # stopping criterion on gradient projected on constraints
    MIN_DELTA_E   : float = 1e-5 # stopping criterion
    MU_MAX        : float = 1e8  # stopping criterion on mu going to +infty
    MU_MIN        : float = 1e-8 # minimal value for mu
    alpha         : float = 0.5  # if iteration is a success, mu = alpha * mu
    beta          : float = 2.   # if iteration fails, mu = beta * mu

@dataclass
class LMVerboseOptions:
    logger_verbose : bool = True  # main verbose for mouette logging
    solver_verbose : bool = False # prints outputs of OSQP solver
    use_tqdm       : bool = True  # progression bar using tqdm
    log_frequency  : int  = 1     # display an info message every 'log_frequency' iterations

@dataclass
class LMFunction:
    name : str # name of the function
    fun  : FunctionType # x -> f(x), J(x) where J is the jacobian
    fun_noJ : FunctionType = None # x -> f(x) without jacobian
    weight : float = 1. # multiplicative weight

    def call(self, X):
        F,J = self.fun(X)
        return self.weight*F, self.weight*J

    def call_noJ(self,X):
        return self.weight*self.fun_noJ(X)

class LevenbergMarquardt(Logger):
    """
    Levenberg-Marquardt algorithm for non-linear least-square minimization.

    References:
        [1] https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
        [2] Constrained Levenberg-Marquardt Method with global complexity bounds, Marumo et al.
    """

    def __init__(self,
        HP : LMParameters = LMParameters(), 
        verbose : LMVerboseOptions = LMVerboseOptions(),
        **kwargs):
        super().__init__("LMopt",verbose.logger_verbose)
        self.verbose_options = verbose
        self.HP = HP

        self.X = None # variables

        ### Linear Constraints
        self._cstM = None
        self._cstL = None
        self._cstU = None

        ### Metric matrix
        self._W : sp.csc_matrix = None

        ### Function to optimize
        self._functions = []

        ### Others
        self._stop_criterion_instance = None # OSQP instance to compute projected gradient norm
        
        ### Additionnal parameters
        self._linsys_solver = kwargs.get("lin_solver", get_osqp_lin_solver())

    def register_constraints(self, A : sp.spmatrix, l : np.ndarray = None, u : np.ndarray = None):
        """
        Adds linear constraints to the optimization problem :

        min_X ||F(X)||² s.t.  l <= AX <= u

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

    def register_function(self, fun, fun_noJ=None, weight:float=1., name:str=None):
        """
        Adds a function to minimize.

        Args:
            fun (python callable): A function taking a single argument X (np.array) returning a vector of values F and a (sparse or dense) jacobian matrix J

            fun_noJ (python callable, optional): The same function that avoids the computation of the jacobian for speed purposes. If not provided, will use the function provided above and ignore the jacobian.

            weight (float, optional): real weight to be applied to the function. Defaults to 1..
            
            name (str, optional): Name of the function in the logs. If not specified, the function will be given a default name.
        """
        if name is None:
            name = f"fun{len(self._functions)}"
        if fun_noJ is None:
            fun_noJ = lambda x : fun(x)[0]
        self._functions.append( LMFunction(name, fun, fun_noJ, weight) )

    def set_metric_matrix(self, W : sp.spmatrix):
        # TODO : some assertions on the matrix
        self._W = W.tocsc()

    def energy(self, X, jac=True, which=None):
        funs_to_call = self._functions if which is None else [fun for fun in self._functions if fun.name in which]
        if jac:
            F,J = [],[]
            for f in funs_to_call:
                fi,ji = f.call(X)
                F.append(fi)
                J.append(ji)
            return F,J
        return [f.call_noJ(X) for f in funs_to_call]

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

        self.X = x_init # variable vector
        
        if self.HP.N_ITER_MAX <= 0 :
            # return current energy
            en = self.energy(x_init, jac=False)
            return 0.5*sum(np.dot(e,e) for e in en)

        if not self._functions: 
            # No functions to optimize
            self.log("No function to optimize.")
            return 0.

        if self._W is None:
            # Metric matrix is Id by default (classical L2 norm)
            self._W = sp.identity(x_init.size, format="csc")

        mu = 1.
        mu_avg = mu
        update = True
        grad_norm = None
        step_norm = 0.
        RelDeltaE = 0.

        # iterable for optimization steps
        if self.verbose_options.logger_verbose and self.verbose_options.use_tqdm:
            print()
            iterobj = trange(self.HP.N_ITER_MAX, total=self.HP.N_ITER_MAX, position=1, leave=False, ncols=100, unit="")
        else:
            iterobj = range(self.HP.N_ITER_MAX)

        try:
            # Iteration loop
            for it in iterobj:                

                if update: # energy only changes if we performed a step last iteration
                    flist, Jlist = self.energy(self.X)
                    Jx = sp.vstack(Jlist).tocsr()
                    fx = np.concatenate(flist)
                    Jt = Jx.transpose()
                    JtJ = Jt.dot(Jx)
                    q = Jt.dot(fx) # gradient

                    Ex = np.dot(fx,fx)/2

                    if Ex<self.HP.ENERGY_MIN:
                        return self._end_optimization(Ex, "Ex < Ex_min") # zero found
                    if np.isnan(Ex):
                        return self._end_optimization(Ex, "NaN value in energy")
                    if np.isposinf(Ex):
                        mu = self.HP.beta*mu
                        update=False
                        continue
                    grad_stop, grad_norm = self._stop_criterion_projgrad(q)
                    if grad_stop:
                        return self._end_optimization(Ex, "projected grad norm on constraints < min_grad_norm")

                if mu>self.HP.MU_MAX:
                    return self._end_optimization(Ex, "mu > mu_max")
                    
                gamma = mu * np.sqrt(2*Ex)
                osqp_instance = OSQP()
                if self._cstM is not None:
                    # OSQP solve inside iteration loop computes the increment x_{n-1} - x_n, so constraints have to be shifted
                    xCst = self._cstM @ self.X
                    cstL = self._cstL - xCst
                    cstU = self._cstU - xCst
                else:
                    cstL,cstU = None, None
                osqp_instance.setup(JtJ + gamma*self._W, q=q, A=self._cstM, l=cstL, u=cstU,
                    verbose=self.verbose_options.solver_verbose,
                    eps_abs=1e-3, eps_rel=1e-3,
                    max_iter=100, polish=True, check_termination=10, 
                    adaptive_rho=True, linsys_solver= self._linsys_solver)
                s = osqp_instance.solve().x # computed increment

                if s[0] is not None:
                    ms = fx + Jx.dot(s)
                    ms = np.dot(ms,ms)/2 + gamma*np.dot(s,s)/2
                    fxs = self.energy(self.X + s, jac=False)
                    fxs = np.concatenate(fxs)
                    Exs = np.dot(fxs,fxs)/2

                update = (s is not None) and (Exs <= ms)  # whether to make a step (True) or increase mu (False)
                if update:
                    RelDeltaE = abs(Exs-Ex)/Ex
                    if RelDeltaE < self.HP.MIN_DELTA_E :
                        return self._end_optimization(Ex, "Relative progression of energy < ΔE_min")
                    self.X += s
                    step_norm = geom.norm(s)
                    if step_norm < self.HP.MIN_STEP_NORM :
                        return self._end_optimization(Ex, "Step norm < min_step_norm")
                    mu, mu_avg = max(self.HP.MU_MIN, self.HP.alpha*mu_avg), mu
                else:
                    mu = self.HP.beta*mu

                if self.verbose_options.logger_verbose and self.verbose_options.log_frequency >0 and it%self.verbose_options.log_frequency==0:
                    energies = [np.dot(_f,_f)/2 for _f in flist]
                    if self.verbose_options.use_tqdm:
                        tqdm_log = prefix
                        if len(self._functions)>1:
                            for n,e in zip([f.name for f in self._functions], energies):
                                tqdm_log+="{}: {:.2E} | ".format(n,e)
                            tqdm_log += "Total: {:.2E} | Grad: {:.2E} | ".format(Ex, grad_norm)
                        else:
                            tqdm_log += "{}: {:.2E} | Grad: {:.2E} | ".format(self._functions[0].name, Ex, grad_norm)
                        tqdm_log += "ΔE {:.2E} | Step {:.2E} | Mu: {:.2E}".format(RelDeltaE, step_norm, mu)
                        tqdm_log += " " * 10
                        tqdm.write(prefix)
                        tqdm.write(tqdm_log)
                    else:
                        log = f"{it+1}/{self.HP.N_ITER_MAX} | "
                        if len(self._functions)>1:
                            for n,e in zip([f.name for f in self._functions], energies):
                                log+="{}: {:.2E} | ".format(n,e)
                            log += "Total: {:.2E} | Grad: {:.2E} | ".format(Ex, grad_norm)
                        else:
                            log += "{}: {:.2E} | Grad: {:.2E} | ".format(self._functions[0].name, Ex, grad_norm)
                        log += "ΔE {:.2E} | Step {:.2E} | Mu: {:.2E}".format(RelDeltaE, step_norm, mu)
                        print(log)
                    
        except KeyboardInterrupt:
            self.log("Manual interruption")
        return self._end_optimization(Ex, "max iteration reached")

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
            if self.verbose_options.use_tqdm: print("\n\r")
        return energy        