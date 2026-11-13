import mouette as M
from data.volumes import *
from utils import *

def test_new():
    m = M.mesh.RawMeshData().instantiate(3)
    assert len(m.vertices)==0
    assert len(m.edges)==0
    assert len(m.faces)==0
    assert len(m.cells)==0

### Io tests ###

@pytest.mark.parametrize("ext", ["mesh", "meshb", "geogram_ascii", "tet", "vtk", "vtu", "msh"])
@pytest.mark.parametrize("v", VOLUMES.items())
def test_io(v, ext, tmp_path):
    name, v = v
    print("INPUT=", name)
    build_test_io(v, tmp_path, ext, 3)