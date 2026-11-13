import mouette as M
from data.surfaces import *
import numpy as np
import pytest

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_translate(s, tmp_path):
    name, s = s
    rand_vec = np.random.random(3)
    p_init = s.vertices[0].copy()
    translated = M.transform.translate(s, rand_vec)
    p_final = translated.vertices[0].copy()
    assert (p_final == rand_vec + p_init).all()


@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_fit_into_cube(s, tmp_path):
    name, s = s
    s2 = M.transform.scale(s, 100*np.random.random())
    s2 = M.transform.translate(s2, 100*M.Vec.random(3))
    s2 = M.transform.fit_into_unit_cube(s2)
    pmin = np.min(s2.vertices._data, axis=0)
    pmax = np.max(s2.vertices._data, axis=0)
    assert np.all(pmin >= M.Vec(0.,0.,0.))
    assert np.all(pmax <= M.Vec(1.,1.,1.))
    

