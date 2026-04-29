import mouette as M
from data.surfaces import *
import numpy as np

@pytest.mark.parametrize("s", [surf_cube()])
def test_translate(s, tmp_path):
    rand_vec = np.random.random(3)
    cube2 = M.transform.translate(s, rand_vec)
    pmin = np.min(cube2.vertices._data, axis=0)
    pmax = np.max(cube2.vertices._data, axis=0)
    assert (pmin == rand_vec).all()
    assert (pmax == rand_vec + M.Vec(1.,1.,1.)).all()


@pytest.mark.parametrize("s", surfaces)
def test_fit_into_cube(s, tmp_path):
    s2 = M.transform.scale(s, 100*np.random.random())
    s2 = M.transform.translate(s2, 100*M.Vec.random(3))
    s2 = M.transform.fit_into_unit_cube(s2)
    pmin = np.min(s2.vertices._data, axis=0)
    pmax = np.max(s2.vertices._data, axis=0)
    assert (pmin >= M.Vec(0.,0.,0.)).all()
    assert (pmax <= M.Vec(1.,1.,1.)).all()
    

