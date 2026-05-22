from __future__ import annotations

from types import SimpleNamespace

from src.simulation.phase_control import (
    _select_best_phase,
    greedy_score_for_phase,
    phase_lane_pairs_from_logic,
    pressure_for_lane_pairs,
)


def _phase(state: str) -> SimpleNamespace:
    return SimpleNamespace(state=state)


def test_phase_lane_pairs_green_links_only():
    phases = [_phase("y"), _phase("G"), _phase("rG")]
    controlled = (
        (("lane_in_0", "lane_out_0", 0),),
        (("lane_in_1", "lane_out_1", 0),),
    )
    pairs = phase_lane_pairs_from_logic(phases, controlled)
    assert pairs[0] == []
    assert pairs[1] == [("lane_in_0", "lane_out_0")]
    assert pairs[2] == [("lane_in_1", "lane_out_1")]


def test_select_best_phase_respects_min_green():
    scores: list[float | None] = [None, 5.0, 10.0]
    assert _select_best_phase(scores, current_phase=1, steps_on_phase=0, min_green_steps=2) == 1
    assert _select_best_phase(scores, current_phase=1, steps_on_phase=2, min_green_steps=2) == 2


def test_pressure_for_lane_pairs_empty():
    assert pressure_for_lane_pairs([]) == 0.0


def test_greedy_score_empty():
    assert greedy_score_for_phase([]) == 0.0
