from mouette.processing import *
from mouette.mesh.datatypes.type_checks import BadMeshTypeException
from mouette.utils.argument_check import InvalidArgumentTypeError,InvalidArgumentValueError, InvalidRangeArgumentError
from data import *

@pytest.mark.parametrize("m",[surf_spline(), surf_half_sphere()])
def test_connection_faces(m):
    conn = SurfaceConnectionFaces(m)
    assert True

@pytest.mark.parametrize("m",[surf_spline(), surf_half_sphere()])
def test_connection_vertices(m):
    conn = SurfaceConnectionVertices(m)
    assert True

def test_surface_framefield_wrong_mesh():
    m = M.mesh.new_polyline()
    try:
        ff = SurfaceFrameField(m, "vertices")
        assert False
    except BadMeshTypeException as e:
        assert True

def test_surface_framefield_wrong_elements():
    m = M.mesh.new_surface()
    try:
        ff = SurfaceFrameField(m, 42)
        assert False
    except InvalidArgumentTypeError as e:
        assert True
    
    try:
        ff = SurfaceFrameField(m,"foo")
        assert False
    except InvalidArgumentValueError as e:
        assert True


def test_surface_frame_field_bad_numerical_values():
    m = M.mesh.new_surface()
    try:
        ff = SurfaceFrameField(m, "vertices", n_smooth="pouet")
        assert False
    except InvalidArgumentTypeError as e:
        assert True

    try:
        ff = SurfaceFrameField(m,"faces",n_smooth=-1)
        assert False
    except InvalidRangeArgumentError as e:
        assert True

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere(), surf_pointy()])
def test_surface_framefield_vertices(m):
    ff = SurfaceFrameField(m, "vertices")()
    ff.flag_singularities()
    assert m.faces.has_attribute("singuls")

@pytest.mark.parametrize("m", [surf_pointy()])
def test_CADFF(m):
    ff = SurfaceFrameField(m, "vertices", cad_correction=True)()
    ff.flag_singularities()
    assert m.faces.has_attribute("singuls")

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere(), surf_pointy()])
def test_surface_framefield_faces(m):
    ff = SurfaceFrameField(m, "faces")()
    ff.flag_singularities()
    assert m.vertices.has_attribute("singuls")

@pytest.mark.parametrize("m", [surf_circle()])
def test_surface_framefield_custom_connection_vertices(m):
    conn = SurfaceConnectionVertices(m)
    ff = SurfaceFrameField(m, "vertices", custom_connection=conn)()
    assert True

@pytest.mark.parametrize("m", [surf_circle()])
def test_surface_framefield_custom_connection_faces(m):
    conn = SurfaceConnectionFaces(m)
    ff = SurfaceFrameField(m, "faces", custom_connection=conn)()
    assert True

@pytest.mark.parametrize("m", [surf_feat()])
def test_surface_framefield_custom_features(m):
    feat = FeatureEdgeDetector()
    ff = SurfaceFrameField(m, "vertices", custom_feature=feat)()
    ff = SurfaceFrameField(m, "faces", custom_feature=feat)()
    assert True

@pytest.mark.parametrize("m", [surf_circle()])
def test_surface_framefield_custom_singus_vertices(m):
    singus = m.faces.create_attribute("singuls", float)
    singus[0] = 1
    singus[10] = 1
    singus[45] = 1
    singus[76] = 1
    ff = SurfaceFrameField(m, "vertices", singularity_indices=singus)()
    assert True

@pytest.mark.parametrize("m", [surf_circle()])
def test_surface_framefield_custom_singus_faces(m):
    singus = m.vertices.create_attribute("singuls", float)
    singus[64] = 1
    singus[80] = 1
    singus[142] = 1
    singus[150] = 1
    ff = SurfaceFrameField(m, "faces", singularity_indices=singus)()
    assert True

@pytest.mark.parametrize("m", [surf_circle()])
def test_surface_framefield_custom_singus_faces_invalid_topo(m):
    singus = m.vertices.create_attribute("singuls", float)
    singus[64] = 1
    singus[80] = -1 # singus do not verify Poincarré-Hopf
    try:
        ff = SurfaceFrameField(m, "faces", singularity_indices=singus)()
        assert False
    except Exception as e:
        assert True

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere(), surf_pointy()])
def test_CurvatureVertices(m):
    ff = PrincipalDirections(m, "vertices")()
    ff.flag_singularities()
    assert m.faces.has_attribute("singuls")

@pytest.mark.parametrize("m", [surf_circle(), surf_spline(), surf_half_sphere(), surf_pointy()])
def test_CurvatureFaces(m):
    ff = PrincipalDirections(m, "faces")()
    ff.flag_singularities()
    assert m.vertices.has_attribute("singuls")