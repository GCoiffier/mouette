import mouette as M

def test_vec_zeros_constructor():
    V = M.Vec.zeros(5)
    assert V.size == 5
    for i in range(5):
        assert V[i] == 0

def test_vec_random_constructor():
    V = M.Vec.random(10)
    assert V.size == 10

def test_vec_from_list():
    lst = [x for x in range(10)]
    V = M.Vec(lst)
    assert V.size==10

def test_vec_xyz_accessor():
    V = M.Vec([1,0,0])
    assert V.x == 1
    assert V.y == 0
    assert V.z == 0

def test_vec_xyz_setters():
    V = M.Vec.zeros(4)
    V.x = 42
    V.y = 64
    V.z = 12
    assert V.x == 42
    assert V[0] == 42
    assert V.y == 64
    assert V[1] == 64
    assert V.z == 12
    assert V[2] == 12
    assert V[3] == 0

def test_vec_bracket_accessor():
    lst = [x for x in range(10)]
    V = M.Vec(lst)
    for i in range(10):
        assert V[i] == i