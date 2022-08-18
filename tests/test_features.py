import pygeomesh as GEO
from data import *
import pytest

@pytest.mark.parametrize("m", surfaces)
def test_feature_detect(m):
    feat = GEO.processing.FeatureEdgeDetector(False,True,False)
    feat.run(m)
    assert isinstance(feat.feature_vertices, set)
    assert isinstance(feat.feature_graph, GEO.mesh.PolyLine)