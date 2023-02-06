from mouette.processing import *
from data import *

@pytest.mark.parametrize("m", [vol_cube()])
def test_volume_framefield_vertices(m):
    ff = VolumeFrameField(m, "vertices")
    ff.run()
    ff.flag_singularities()
    assert m.faces.has_attribute("singuls")

@pytest.mark.parametrize("m", [vol_cube()])
def test_volume_framefield_cells(m):
    ff = VolumeFrameField(m, "cells")
    ff.run()
    ff.flag_singularities()
    assert m.vertices.has_attribute("singuls")
    assert m.edges.has_attribute("singuls")
