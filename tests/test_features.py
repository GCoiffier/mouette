import mouette as M
from data import *
import pytest

@pytest.mark.parametrize("m", surfaces)
def test_feature_detect(m):
    feat = M.processing.FeatureEdgeDetector(only_border= False, flag_corners=True)
    feat.run(m)
    assert isinstance(feat.feature_vertices, set)
    assert isinstance(feat.feature_graph, M.mesh.PolyLine)