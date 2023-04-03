from mouette.processing import parametrization as PARAM
from utils import *
from data import *

########## LSCM ##########

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere()])
def test_LSCM(m):
    lscm = PARAM.LSCM(m, verbose=False, eigen=True, save_on_corners=True, solver_verbose=False)
    lscm.run()
    assert m.face_corners.has_attribute("uv_coords")

    lscm = PARAM.LSCM(m, verbose=False, eigen=True, save_on_corners=False, solver_verbose=False)
    lscm.run()
    assert m.vertices.has_attribute("uv_coords")

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere()])
def test_LSCM_no_eigen(m):
    lscm = PARAM.LSCM(m, verbose=False, eigen=False, save_on_corners=True, solver_verbose=False)
    lscm.run()
    assert m.face_corners.has_attribute("uv_coords")
    
    lscm = PARAM.LSCM(m, verbose=False, eigen=False, save_on_corners=False, solver_verbose=False)
    lscm.run()
    assert m.vertices.has_attribute("uv_coords")

@pytest.mark.parametrize("m", [surf_cube()])
def test_LSCM_not_disk(m):
    try:
        _ = PARAM.LSCM(m,verbose=False, eigen=False, save_on_corners=False)()
        assert False
    except:
        assert True

@pytest.mark.parametrize("m", [surf_half_sphere()])
def test_export_LSCM(m, tmp_path):
    _ = PARAM.LSCM(m, verbose=False, eigen=False, save_on_corners=True, solver_verbose=False)()
    build_test_io(m, tmp_path, "obj", 2)
    build_test_io(m, tmp_path, "geogram_ascii", 2)

########## Tutte Embedding ##########

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere()])
def test_tutte_circle(m):
    tutte = PARAM.TutteEmbedding(m,verbose=False, save_on_corners=True, boundary_mode="circle")
    tutte.run()
    assert m.face_corners.has_attribute("uv_coords")

    tutte = PARAM.TutteEmbedding(m,verbose=False, save_on_corners=False, boundary_mode="circle")
    tutte.run()
    assert m.vertices.has_attribute("uv_coords")

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere()])
def test_tutte_square(m):
    tutte = PARAM.TutteEmbedding(m,verbose=False, save_on_corners=True, boundary_mode="square")
    tutte.run()
    assert m.face_corners.has_attribute("uv_coords")

    tutte = PARAM.TutteEmbedding(m,verbose=False, save_on_corners=False, boundary_mode="square")
    tutte.run()
    assert m.vertices.has_attribute("uv_coords")

@pytest.mark.parametrize("m", [surf_circle()])
def test_tutte_invalid_boundary(m):
    try:
        tutte = PARAM.TutteEmbedding(m,verbose=False, save_on_corners=True, boundary_mode="foo")
        tutte.run()
        assert False
    except M.utils.InvalidArgumentValueError:
        assert True
        
########## Param Distortion ##########

@pytest.mark.parametrize("m", [surf_half_sphere()])
def test_param_distortion(m):
    _ = PARAM.LSCM(m, verbose=False, eigen=False, save_on_corners=True, solver_verbose=False)()
    disto = PARAM.ParamDistortion(m, "uv_coords", save_on_mesh=True, verbose=False)
    disto.run()

    # check mesh attributes
    for attr_name in ("conformal_dist", "scale_dist", "stretch_dist", "shear_dist", "iso_dist", "det"):
        assert m.faces.has_attribute(attr_name)

    # check summary
    for key in ("conformal", "iso", "shear", "scale", "stretch_mean", "stretch_max"):
        assert key in disto.summary

@pytest.mark.parametrize("m", [surf_uv_sphere_quads()])
def test_quad_quality(m):
    disto = PARAM.QuadQuality(m, save_on_mesh=True, verbose=False)
    disto.run()
    for attr_name in ("conformal_dist", "scale_dist", "stretch_dist", "det"):
        assert m.faces.has_attribute(attr_name)

    # check summary
    for key in ("conformal", "det", "scale", "stretch"):
        assert key in disto.summary