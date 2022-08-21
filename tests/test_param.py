from mouette.processing import parametrization as PARAM
from utils import *
from data import *

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere()])
def test_LSCM(m, tmp_path):
    lscm = PARAM.LSCM(m,False)
    lscm.run(True, True, True, False)

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere()])
def test_LSCM_no_eigen(m, tmp_path):
    lscm = PARAM.LSCM(m,False)
    lscm.run(False, True, True, False)

@pytest.mark.parametrize("m", [surf_half_sphere()])
def test_export_LSCM(m, tmp_path):
    lscm = PARAM.LSCM(m,False)(True, True, True, False)
    build_test_io(m, tmp_path, "obj", 2)
    build_test_io(m, tmp_path, "geogram_ascii", 2)