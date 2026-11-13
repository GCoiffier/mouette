import mouette as M
from data import *
import pytest

@pytest.mark.parametrize("s", ALL_SURFACES.items())
def test_feature_detect(s):
    _,m = s
    feat = M.processing.FeatureEdgeDetector(only_border= False, flag_corners=True)
    feat.run(m)
    assert isinstance(feat.feature_vertices, set)
    assert isinstance(feat._feature_graph, M.mesh.PolyLine)

@pytest.mark.parametrize("m", [ALL_SURFACES["spline"], ALL_SURFACES["half_sphere"]])
def test_feature_detect_only_border(m):
    feat = M.processing.FeatureEdgeDetector(only_border=True, flag_corners=False, compute_feature_graph=False, verbose=False)
    feat.run(m)
    for v in feat.feature_vertices:
        assert m.is_vertex_on_border(v)
    for e in feat.feature_edges:
        assert m.is_edge_on_border(*m.edges[e])