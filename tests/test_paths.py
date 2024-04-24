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