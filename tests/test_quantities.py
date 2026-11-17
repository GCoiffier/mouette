import mouette as M
from utils import *
from data import *

def _make_test_quantity(mesh, attr):
    val = attr(mesh)
    val = attr(mesh, persistent=False)

##### Vertices 

@pytest.mark.parametrize("s", ALL_SURFACES.items())
@pytest.mark.parametrize("attr_fun", [
    M.attributes.degree,
    M.attributes.angle_defects,
    M.attributes.vertex_normals,
    M.attributes.border_normals,
    M.attributes.edge_length,
    M.attributes.edge_middle_point,
    M.attributes.curvature_matrices,
    M.attributes.cotan_weights,
    M.attributes.face_area,
    M.attributes.face_normals,
    M.attributes.face_barycenter,
    M.attributes.face_circumcenter,
    M.attributes.face_near_border,
    M.attributes.corner_angles,
    M.attributes.cotangent,
    M.attributes.parallel_transport_curvature
])
def test_surface_compatible_attributes(s, attr_fun):
    name,mesh =s
    _make_test_quantity(mesh, attr_fun)
    assert True


@pytest.mark.parametrize("s", TRI_SURFACES.items())
@pytest.mark.parametrize("attr_fun", [
    M.attributes.triangle_aspect_ratio,
])
def test_triangulated_surface_compatible_attributes(s, attr_fun):
    name, mesh = s
    _make_test_quantity(mesh, attr_fun)
    assert True


##### Cells

@pytest.mark.parametrize("s", VOLUMES.items())
@pytest.mark.parametrize("attr_fun", [
    M.attributes.cell_volume,
    M.attributes.cell_barycenter,
    M.attributes.cell_faces_on_boundary
])
def test_surface_compatible_attributes(s, attr_fun):
    name,mesh = s
    _make_test_quantity(mesh, attr_fun)
    assert True


##### Global quantities 

@pytest.mark.parametrize("mesh,char", [
    (ALL_SURFACES["circle"], 1),
    (ALL_SURFACES["cube_triangulated"], 2),
    (ALL_SURFACES["cube_quad"],  2),
    (ALL_SURFACES["torus_triangulated"], 0) 
    ])
def test_euler_characteristic(mesh, char):
    assert M.attributes.euler_characteristic(mesh) == char

@pytest.mark.parametrize("s", TRI_SURFACES.items())
def test_gauss_bonnet_theorem(s):
    # sum of angle defects is equal to 2*pi times the euler characteristic 
    name,mesh = s
    ang = M.attributes.angle_defects(mesh, persistent=False)
    sum_ang = np.sum(ang)
    print(ang, sum_ang)
    charac = 2*np.pi*M.attributes.euler_characteristic(mesh)
    assert np.isclose(sum_ang, charac)

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_mean_edge_length(s):
    name,mesh = s
    m = M.attributes.mean_edge_length(mesh)
    assert True

@pytest.mark.parametrize("s", TRI_SURFACES.items())
def test_mean_face_area(s):
    name,mesh = s
    m = M.attributes.mean_face_area(mesh)
    assert True

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_total_area(s):
    name,mesh = s
    m = M.attributes.total_area(mesh)
    assert True

@pytest.mark.parametrize("s", VOLUMES.items())
def test_mean_cell_volume(s):
    name,mesh = s
    m = M.attributes.mean_cell_volume(mesh)
    assert True

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_barycenter(s):
    name,mesh = s
    m = M.attributes.barycenter(mesh)
    assert True