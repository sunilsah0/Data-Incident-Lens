import networkx as nx

from engine.incidents import group_alerts, peel_roots, rank_root_causes
from experiments.lab import propagate


def chain():
    g = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("a", "d")])
    return g


def test_connected_alerts_form_one_incident():
    assert len(group_alerts(chain(), ["a", "c"])) == 1


def test_unconnected_alerts_stay_separate():
    g = chain()
    g.add_edge("x", "y")
    assert len(group_alerts(g, ["c", "y"])) == 2


def test_most_upstream_wins_when_everything_alerts():
    ranked = rank_root_causes(chain(), {"a": 0.8, "b": 0.8, "c": 0.8, "d": 0.8})
    assert ranked[0]["asset"] == "a"


def test_downstream_victim_is_not_blamed():
    ranked = rank_root_causes(chain(), {"b": 0.9, "c": 0.9})
    assert ranked[0]["asset"] == "b"


def test_peeling_finds_two_independent_roots():
    g = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("x", "y")])
    roots = peel_roots(g, {"a": 0.9, "b": 0.9, "x": 0.9, "y": 0.9})
    assert set(roots) == {"a", "x"}


def test_unused_column_is_not_affected():
    assert propagate("SELECT a, b FROM t WHERE c > 0", {"a"}) == {"a"}


def test_filter_column_hits_every_output():
    assert propagate("SELECT a, b FROM t WHERE c > 0", {"c"}) == {"a", "b"}