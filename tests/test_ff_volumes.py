from mouette import framefield as ff
from data import *

@pytest.mark.parametrize("m", [vol_cube()])
def test_volume_framefield_vertices(m):
    field = ff.VolumeFrameField(m, "vertices")
    field.run()
    field.flag_singularities()
    assert m.faces.has_attribute("singuls")

@pytest.mark.parametrize("m", [vol_cube()])
def test_volume_framefield_cells(m):
    field = ff.VolumeFrameField(m, "cells")
    field.run()
    field.flag_singularities()
    assert m.vertices.has_attribute("singuls")
    assert m.edges.has_attribute("singuls")
