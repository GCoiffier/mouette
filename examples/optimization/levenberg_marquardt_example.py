import mouette as M
import numpy as np
import scipy.sparse as sp

def function(x, a, b):
    # Rosenbrock function
    return np.array([(a-x[0])**2 + b*(x[1] - x[0]**2)**2])

def function_and_jac(x, a, b):
    # Rosenbrock function and Jacobian
    f = function(x,a,b)
    J = sp.csr_matrix([2*(x[0]-a) - 4*b*x[0]*(x[1]-x[0]*x[0]),  2*b*(x[1]-x[0]*x[0])]).reshape((1,2))
    return f,J

if __name__== "__main__":
    A = 1.
    B = 2.

    options = M.optimize.levenberg_marquardt.LMParameters(
        N_ITER_MAX    = 1000, # stopping criterion on maximal number of iterations
        ENERGY_MIN    = 1E-19, # stopping criterion for reaching zero
        MIN_STEP_NORM = 0., # stopping criterio, on ||x_n - x_{n-1}||
        MIN_GRAD_NORM = 0., # stopping criterion on gradient projected on constraints
        MIN_DELTA_E   = 0., # stopping criterion
        MU_MAX        = 1e8 , # stopping criterion on mu going to +infty
        MU_MIN        = 1e-8, # minimal value for mu
        alpha         = 0.5 , # if iteration is a success, mu = alpha * mu
        beta          = 2.    # if iteration fails, mu = beta * mu
    )

    verbose_options = M.optimize.levenberg_marquardt.LMVerboseOptions(
        logger_verbose=True,
        solver_verbose=False,
        use_tqdm=False,
        log_frequency=1
    )

    optimizer = M.optimize.LevenbergMarquardt(options, verbose_options)
    optimizer.register_function(lambda x : function_and_jac(x, A, B), lambda x : function(x, A, B), name="Rosenbrock")
    optimizer.run(x_init=np.random.random((2,)))

    print("Local minimum found:" , optimizer.X)
    print("Expected minimum :", [A,A])