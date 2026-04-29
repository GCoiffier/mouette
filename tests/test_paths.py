import mouette as M
from data import *

@pytest.mark.parametrize("s",[
    surf_half_sphere(),
])
def test_shortest_path(s):
    path, polyline = M.processing.shortest_path(s, 57, [82], export_path_mesh=True)
    assert True

@pytest.mark.parametrize("s",[
    surf_half_sphere(),
])
def test_shortest_path_border(s):
    path, polyline = M.processing.shortest_path_to_border(s, 59, weights="one",export_path_mesh=True)
    assert True


@pytest.mark.parametrize("m", [surf_circle()])
def test_n_closest(m):
    SOURCE, TARGET1, TARGET2, TARGET3, TARGET4 = 59, 219, 124, 206, 0
    paths = M.processing.closest_n_vertices(m, SOURCE, 2, [TARGET1, TARGET2,TARGET3, TARGET4])
    assert len(paths)==2
    assert TARGET1 in paths
    assert TARGET2 in paths
    assert TARGET3 not in paths
    assert TARGET4 not in paths