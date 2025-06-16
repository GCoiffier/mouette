import mouette as M
from mouette.processing import parametrization as PARAM
from utils import *
from data import *
from math import pi

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


########## Boudary First Flattenning ##########

@pytest.mark.parametrize("mesh", [surf_circle(), surf_spline(), surf_half_sphere()])
def test_bff_boundary_scale(mesh : M.mesh.SurfaceMesh):
    bnd_scale = mesh.vertices.create_attribute("scale", float, default_value=1.)
    param = PARAM.BoundaryFirstFlattening(mesh, bnd_scale_fctr=bnd_scale, verbose=True, save_on_corners=False)
    mesh = param.run() # vertex ordering has changed
    assert mesh.vertices.has_attribute("uv_coords")

@pytest.mark.parametrize("mesh", [surf_circle(), surf_spline(), surf_half_sphere()])
def test_bff_boundary_curv(mesh : M.mesh.SurfaceMesh):
    curv = 2*pi/len(mesh.boundary_vertices)
    bnd_curv = mesh.vertices.create_attribute("curv", float, default_value=curv)
    param = PARAM.BoundaryFirstFlattening(mesh, bnd_curvature=bnd_curv, verbose=True, save_on_corners=True)
    mesh = param.run()
    assert mesh.face_corners.has_attribute("uv_coords")

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


########## FRAME FIELD INTEGRATION ##########

@pytest.mark.parametrize("m", [surf_circle(), surf_half_sphere(), surf_spot()])
def test_FF_integrate(m):
    ff = M.framefield.SurfaceFrameField(m, "faces", verbose=True, features=False)
    ff_param = M.parametrization.FrameFieldIntegration(ff, scaling=10., verbose=True)
    ff_param.run()
    assert m.vertices.has_attribute("singuls")
    assert m.face_corners.has_attribute("uv_coords")
    assert isinstance(ff_param.flat_mesh, M.mesh.SurfaceMesh)

@pytest.mark.parametrize("m", [surf_circle(), surf_feat(), surf_cube_subdiv()])
def test_FF_integrate_with_features(m):
    ff = M.framefield.SurfaceFrameField(m, "faces", verbose=True, features=True)()
    ff_param = M.parametrization.FrameFieldIntegration(ff, scaling=10., verbose=True)
    ff_param.run()
    assert m.vertices.has_attribute("singuls")
    assert m.face_corners.has_attribute("uv_coords")
    assert isinstance(ff_param.flat_mesh, M.mesh.SurfaceMesh)

@pytest.mark.parametrize("m", [surf_half_sphere(), surf_torus(), surf_spot()])
def test_FF_integrate_curvature(m):
    ff = M.framefield.PrincipalDirections(m, "faces", verbose=True, features=False)
    ff_param = M.parametrization.FrameFieldIntegration(ff, scaling=10., verbose=True)
    ff_param.run()
    assert m.vertices.has_attribute("singuls")
    assert m.face_corners.has_attribute("uv_coords")
    assert isinstance(ff_param.flat_mesh, M.mesh.SurfaceMesh)


########## CONE PARAMETRIZATION ##########

@pytest.mark.parametrize("mesh,cones", [
    (surf_circle(), [(68, pi/2), (263,pi), (251, pi/2), (333, pi/2), (59, -pi/2)]),
    (surf_spot(), [(702, pi/2),(957, pi/2), (895, pi/2), (995,pi/2), (553,pi/2), (769, pi/2), (752, pi/2), (640, pi/2)]),
    (surf_torus(), [])
])
def test_cone_param(mesh : M.mesh.SurfaceMesh, cones):
    cones_attr = mesh.vertices.create_attribute("cones", float)
    for (vid,val) in cones:
        cones_attr[vid] = val
    param = M.parametrization.ConformalConeParametrization(mesh, cones_attr, verbose=True)
    param.run()
    assert mesh.face_corners.has_attribute("uv_coords")
    assert isinstance(param.flat_mesh, M.mesh.SurfaceMesh)


@pytest.mark.parametrize("mesh", [surf_spot()])
def test_cone_param_invalid_cones(mesh : M.mesh.SurfaceMesh):
    cone_attr = mesh.vertices.create_attribute("cones", float)
    cone_attr[0] = 0.1 
    try:
        param = M.parametrization.ConformalConeParametrization(mesh, cone_attr)
        param.run()
        assert False
    except M.parametrization.ConformalConeParametrization.InvalidConesException as e:
        assert True 