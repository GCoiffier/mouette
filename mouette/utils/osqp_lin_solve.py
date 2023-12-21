from osqp import OSQP
from scipy.sparse import identity

def get_osqp_lin_solver():
    try:
        inst = OSQP()
        inst.setup(P=identity(1, format="csc"), verbose=False, linsys_solver="mkl pardiso")
        return "mkl pardiso"
    except ValueError:
        return "qdldl"