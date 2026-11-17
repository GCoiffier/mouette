import pytest
import mouette as M
import numpy as np
from data.polylines import *
from utils import *

def _reference(mesh):
    """Expected connectivity, computed in pure python from the edges stored in the mesh.

    Returns:
        adj (list of sets): adj[v] is the set of neighbors of vertex v
        eid (dict): (min(a,b), max(a,b)) -> index of edge (a,b)
    """
    adj = [set() for _ in range(len(mesh.vertices))]
    eid = dict()
    for i, (a,b) in enumerate(np.asarray(mesh.edges).tolist()):
        adj[a].add(b)
        adj[b].add(a)
        eid[M.utils.keyify(a,b)] = i
    return adj, eid

def test_new():
    m = M.mesh.RawMeshData().instantiate(1)
    assert len(m.vertices)==0
    assert len(m.edges)==0

def test_no_edges():
    raw = M.mesh.RawMeshData()
    raw.vertices = np.random.random((4,3))
    m = raw.instantiate(1)
    for v in m.id_vertices:
        assert m.connectivity.vertex_to_vertices(v) == []
        assert m.connectivity.vertex_to_edges(v) == []
    assert m.connectivity.edge_id(0,1) is None

### Io tests ###

@pytest.mark.parametrize("ext", ["geogram_ascii", "mesh", "vtk"])
@pytest.mark.parametrize("p", ALL_POLYLINES.items())
def test_io_polyline(p, ext, tmp_path):
    name, p = p
    build_test_io(p, tmp_path, ext, 1)

### Hand-checked connectivity ###

def test_adj():
    p = POLYLINES["polyline1"] # edges (0,1), (1,2)
    assert sorted(p.connectivity.vertex_to_vertices(1)) == [0,2]
    assert sorted(p.connectivity.vertex_to_edges(1)) == [0,1]
    assert p.connectivity.vertex_to_vertices(0) == [1]
    assert p.connectivity.edge_id(2,1) == 1
    assert p.connectivity.edge_id(0,2) is None

def test_other_edge_end_one_edge():
    p = POLYLINES["one_edge"]
    assert p.connectivity.other_edge_end(0,0) == 1
    assert p.connectivity.other_edge_end(0,1) == 0

def test_adj_isolated():
    p = POLYLINES_CONNECTIVITY["isolated"]
    for v in (0,4,8):
        assert p.connectivity.vertex_to_vertices(v) == []
        assert p.connectivity.vertex_to_edges(v) == []
    assert sorted(p.connectivity.vertex_to_vertices(6)) == [5,7]

### Return types: the 1.3 contract (lists of python ints, int or None) ###

@pytest.mark.parametrize("p", ALL_POLYLINES.items())
def test_return_types(p):
    name, p = p
    C = p.connectivity
    for v in p.id_vertices:
        vv = C.vertex_to_vertices(v)
        ve = C.vertex_to_edges(v)
        assert type(vv) is list
        assert type(ve) is list
        print(vv)
        assert all([type(x) is int for x in vv])
        assert all([type(x) is int for x in ve])
    a,b = np.asarray(p.edges).tolist()[0]
    assert type(C.edge_id(a,b)) is int
    assert C.edge_id(a,a) is None

### Correctness against a pure python reference ###

@pytest.mark.parametrize("p", ALL_POLYLINES.items())
def test_V2V(p):
    name, p = p
    adj, _ = _reference(p)
    for v in p.id_vertices:
        vv = [int(x) for x in p.connectivity.vertex_to_vertices(v)]
        assert set(vv) == adj[v]
        assert len(vv) == len(adj[v]) # no duplicates

@pytest.mark.parametrize("p", ALL_POLYLINES.items())
def test_V2E(p):
    name, p = p
    adj, eid = _reference(p)
    E = np.asarray(p.edges).tolist()
    for v in p.id_vertices:
        ve = [int(x) for x in p.connectivity.vertex_to_edges(v)]
        assert set(ve) == {eid[M.utils.keyify(v,w)] for w in adj[v]}
        assert len(ve) == len(adj[v])
        for e in ve:
            assert v in E[e]

@pytest.mark.parametrize("p", ALL_POLYLINES.items())
def test_V2E_aligned_with_V2V(p):
    # i-th edge around v is the edge between v and its i-th neighbor
    name, p = p
    C = p.connectivity
    for v in p.id_vertices:
        vv = list(C.vertex_to_vertices(v))
        ve = list(C.vertex_to_edges(v))
        assert len(vv) == len(ve)
        for w,e in zip(vv,ve):
            assert e == C.edge_id(v,w)

@pytest.mark.parametrize("p", ALL_POLYLINES.items())
def test_edge_id_both_orientations(p):
    name, p = p
    C = p.connectivity
    for i,(a,b) in enumerate(np.asarray(p.edges).tolist()):
        assert C.edge_id(a,b) == i
        assert C.edge_id(b,a) == i

@pytest.mark.parametrize("p", ALL_POLYLINES.items())
def test_edge_id_missing(p):
    name, p = p
    adj, eid = _reference(p)
    C = p.connectivity
    n = len(p.vertices)
    rng = np.random.default_rng(42)
    for a,b in rng.integers(0, n, (200,2)).tolist():
        if M.utils.keyify(a,b) in eid: continue
        assert C.edge_id(a,b) is None
        assert C.edge_id(b,a) is None
    for v in p.id_vertices:
        assert C.edge_id(v,v) is None
        if len(adj[v])==0: # isolated vertex
            for w in range(n):
                assert C.edge_id(v,w) is None

@pytest.mark.parametrize("p", ALL_POLYLINES.items())
def test_other_edge_end(p):
    name, p = p
    C = p.connectivity
    n = len(p.vertices)
    for e,(a,b) in enumerate(np.asarray(p.edges).tolist()):
        assert C.other_edge_end(e,a) == b
        assert C.other_edge_end(e,b) == a
        not_on_edge = next((w for w in range(n) if w not in (a,b)), None)
        if not_on_edge is not None:
            assert C.other_edge_end(e, not_on_edge) is None

@pytest.mark.parametrize("p", ALL_POLYLINES.items())
def test_edge_to_vertices(p):
    name, p = p
    for e,(a,b) in enumerate(np.asarray(p.edges).tolist()):
        assert [int(x) for x in p.connectivity.edge_to_vertices(e)] == [a,b]

@pytest.mark.parametrize("p", ALL_POLYLINES.items())
def test_handshake(p):
    name, p = p
    deg = sum(len(p.connectivity.vertex_to_vertices(v)) for v in p.id_vertices)
    assert deg == 2*len(p.edges)

### Laziness and invalidation (fresh meshes, since these tests query in a specific order or mutate) ###

@pytest.mark.parametrize("builder", [pl_sample1, pl_chain, pl_random_graph, pl_hubs, pl_isolated_vertices, pl_reversed_edges])
def test_edge_id_before_v2v(builder):
    p = builder()
    _, eid = _reference(p)
    for i,(a,b) in enumerate(np.asarray(p.edges).tolist()):
        assert p.connectivity.edge_id(b,a) == i

@pytest.mark.parametrize("builder", [pl_sample1, pl_random_graph, pl_isolated_vertices])
def test_clear_rebuilds(builder):
    p = builder()
    adj, eid = _reference(p)
    C = p.connectivity
    C.vertex_to_vertices(0)
    C.edge_id(*np.asarray(p.edges).tolist()[0])
    C.clear()
    for i,(a,b) in enumerate(np.asarray(p.edges).tolist()):
        assert C.edge_id(a,b) == i
    for v in p.id_vertices:
        assert set(int(x) for x in C.vertex_to_vertices(v)) == adj[v]
        assert set(int(x) for x in C.vertex_to_edges(v)) == {eid[M.utils.keyify(v,w)] for w in adj[v]}

def test_clear_after_edit():
    p = pl_sample1() # edges (0,1), (1,2)
    C = p.connectivity
    assert C.edge_id(0,1) == 0
    assert sorted(C.vertex_to_vertices(1)) == [0,2]
    p.edges.set_data(np.array([[0,2]]))
    C.clear()
    assert C.edge_id(0,1) is None
    assert C.edge_id(2,0) == 0
    assert C.vertex_to_vertices(0) == [2]
    assert C.vertex_to_vertices(1) == []
    assert C.vertex_to_edges(2) == [0]
