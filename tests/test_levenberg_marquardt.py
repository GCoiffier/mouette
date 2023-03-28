import pytest
import mouette as M
import numpy as np
import scipy.sparse as sp

@pytest.fixture
def optim():
    opt = M.optimize.LevenbergMarquardt()
    opt.verbose_options.logger_verbose = False
    return opt

def test_LM1(optim):
    optim.register_function(
        lambda X : (2*X, 2*sp.eye(X.size, dtype=np.float32)),
        lambda X : 2*X,
        0.5,
        "test1")
    X0 = np.full(100, 1.)
    en = optim.run(X0)
    assert en< optim.HP.ENERGY_MIN
    assert np.all( abs(optim.X)<1e-6)

def test_LM_only1fun(optim):
    optim.register_function(
        lambda X : (2*X, 2*sp.eye(X.size, dtype=np.float32)),
        weight=1,
        name="test2")
    X0 = np.full(42, 1.)
    en = optim.run(X0)
    assert en< optim.HP.ENERGY_MIN
    assert np.all( abs(optim.X)<1e-6)

def test_LM_constraints1(optim):
    optim.register_function(
        lambda X : (2*X, 2*sp.eye(X.size, dtype=np.float32)),
        weight=1,
        name="test2")
    
    A = np.zeros((2,42))
    A[0,0]  = 1
    A[1,16] = 1
    A = sp.csc_matrix(A)
    B = np.zeros(2) #x0 = x15 = 0
    optim.register_constraints(A,B,B)
    en = optim.run(x_init=np.full(42, 1.))
    print(en, optim.X)
    assert abs(en)< 1e-4
    assert abs(optim.X[0])<1e-4
    assert abs(optim.X[16])<1e-4

def test_LM_constraints2(optim):
    optim.register_function(
        lambda X : (X, sp.eye(X.size, dtype=np.float32)),
        weight=1,
        name="test2")
    A = np.zeros((3,42))
    A[0,0]  =  1 # x0  = 1
    A[1,16] =  1 # x15 = 3
    A[2,16] =  1 
    A[2,20] =  1 # x15 + x19 = 0
    A = sp.csc_matrix(A)
    B = np.array([1,2,0])
    optim.register_constraints(A,B,B)
    en = optim.run(x_init=np.full(42, 1.))
    assert abs(en-4.5)<1e-4
    assert abs(optim.X[0]-1)<1e-4
    assert abs(optim.X[16]-2)<1e-4
    assert abs(optim.X[20]+2)<1e-4

def test_LM2(optim):
    optim.register_function(
        lambda X : (np.array([np.dot(X,X)-1]), sp.csr_matrix(2*X).reshape((1,X.size))),
        lambda X : np.array([np.dot(X,X)-1]),
        1,
        "test3")
    X0 = np.array([1/x for x in range(1,30)])
    en = optim.run(X0)
    assert en< optim.HP.ENERGY_MIN


def test_LM_twofunctions(optim):
    optim.register_function(
        lambda X : (X, sp.eye(X.size, dtype=np.float32)),
        weight=1,
    )
    optim.register_function(
        lambda X : (2*X, 2*sp.eye(X.size, dtype=np.float32)),
        weight=1.2
    )
    en = optim.run(x_init=np.full(42, 1.))
    assert en < optim.HP.ENERGY_MIN  
