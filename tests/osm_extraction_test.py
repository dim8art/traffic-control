import networkx as nx
import pandas as pd

from src.map_engine.extractor import MapExtractor


def test_get_adjacency_list_for_signals():
    extractor = MapExtractor(location=(59.9, 30.3), dist=200)
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)

    extractor.graph = graph
    extractor.signals = pd.DataFrame(index=[1, 2])

    adj = extractor.get_adjacency_list()

    assert 1 in adj
    assert 2 in adj
    assert set(adj[1]) == {2, 3}
    assert set(adj[2]) == {4}