from mouette.processing import framefield as GEOFF
from data import *

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere(), surf_pointy()])
def test_FF2DVertices(m):
    ff = GEOFF.FrameField2DVertices(m, 4, False, False)()
    ff.flag_singularities()
    assert m.faces.has_attribute("singuls")

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere(), surf_pointy()])
def test_CadFF2DVertices(m):
    ff = GEOFF.CadFF2DVertices(m, 4, False)()
    ff.flag_singularities()
    assert m.faces.has_attribute("singuls")

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere(), surf_pointy()])
def test_CurvatureFFVertices(m):
    ff = GEOFF.CurvatureVertices(m, False)()
    ff.flag_singularities()
    assert m.faces.has_attribute("singuls")