import mouette as M
from data.volumes import *
from utils import *

def test_new():
    m = M.mesh.new_volume()
    assert len(m.vertices)==0
    assert len(m.edges)==0
    assert len(m.faces)==0
    assert len(m.cells)==0

### Io tests ###

# @pytest.mark.parametrize("v", volumes)
# def test_io_medit(v, tmp_path):
#     build_test_io(v, tmp_path, "mesh", 3)

# @pytest.mark.parametrize("v", volumes)
# def test_io_geogram_ascii(v, tmp_path):
#     build_test_io(v, tmp_path, "geogram_ascii", 3)

# @pytest.mark.parametrize("v", volumes)
# def test_io_tet(v, tmp_path):
#     build_test_io(v, tmp_path, "tet", 3)