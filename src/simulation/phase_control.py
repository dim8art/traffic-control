"""
Классические адаптивные контроллеры светофоров через TraCI (режим use_phase_actions в env).

- MaxPressure (RESCO, LibSignal): max Σ (очередь_въезд − очередь_выезд) по связям фазы.
- Greedy (RESCO): max Σ (очередь + подъезжающие) на въездных полосах фазы.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import traci


def _lane_halting_count(lane_id: str) -> float:
    try:
        return float(traci.lane.getLastStepHaltingNumber(lane_id))
    except traci.TraCIException:
        return 0.0


def _lane_vehicle_count(lane_id: str) -> float:
    try:
        return float(traci.lane.getLastStepVehicleNumber(lane_id))
    except traci.TraCIException:
        return 0.0


def phase_lane_pairs_from_logic(
    phases: list,
    controlled_links: tuple,
) -> list[list[tuple[str, str]]]:
    """Для каждой фазы — пары (въезд, выезд) с зелёным сигналом."""
    out: list[list[tuple[str, str]]] = []
    for phase in phases:
        pairs: list[tuple[str, str]] = []
        state = phase.state
        for i, ch in enumerate(state):
            if ch not in ("G", "g"):
                continue
            if i >= len(controlled_links):
                continue
            for link in controlled_links[i]:
                if len(link) < 2:
                    continue
                inl, outl = link[0], link[1]
                if inl and outl:
                    pairs.append((inl, outl))
        out.append(pairs)
    return out


def build_phase_lane_pairs(tls_id: str) -> list[list[tuple[str, str]]]:
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    controlled = traci.trafficlight.getControlledLinks(tls_id)
    return phase_lane_pairs_from_logic(logic.phases, controlled)


def pressure_for_lane_pairs(pairs: list[tuple[str, str]]) -> float:
    total = 0.0
    for inl, outl in pairs:
        total += _lane_halting_count(inl) - _lane_halting_count(outl)
    return total


def greedy_score_for_phase(pairs: list[tuple[str, str]]) -> float:
    """Очередь + подъезжающие на уникальных въездах фазы (RESCO Greedy)."""
    in_lanes = {inl for inl, _ in pairs}
    score = 0.0
    for inl in in_lanes:
        halt = _lane_halting_count(inl)
        total = _lane_vehicle_count(inl)
        score += halt + max(0.0, total - halt)
    return score


def _select_best_phase(
    scores: list[float | None],
    *,
    current_phase: int,
    steps_on_phase: int,
    min_green_steps: int,
) -> int:
    if steps_on_phase < min_green_steps:
        return current_phase

    best_phase = current_phase
    best_score: float | None = None

    for phase_idx, raw in enumerate(scores):
        if raw is None:
            continue
        p = float(raw)
        if best_score is None or p > best_score:
            best_score = p
            best_phase = phase_idx
        elif p == best_score and phase_idx == current_phase:
            best_phase = current_phase

    return best_phase


class PhaseController(Protocol):
    def reset(self, tls_ids: list[str]) -> None: ...
    def compute_actions(self, tls_ids: list[str]) -> dict[str, int]: ...


@dataclass
class _PhaseControllerBase:
    min_green_steps: int = 1
    _pairs_by_tls: dict[str, list[list[tuple[str, str]]]] | None = None
    _current_phase: dict[str, int] | None = None
    _steps_on_phase: dict[str, int] | None = None

    def _ensure_ready(self) -> None:
        if self._pairs_by_tls is None or self._current_phase is None or self._steps_on_phase is None:
            raise RuntimeError(
                f"{type(self).__name__}.reset() нужно вызвать до compute_actions()"
            )

    def reset(self, tls_ids: list[str]) -> None:
        self._pairs_by_tls = {}
        self._current_phase = {}
        self._steps_on_phase = {}
        for tls_id in tls_ids:
            self._pairs_by_tls[tls_id] = build_phase_lane_pairs(tls_id)
            cur = int(traci.trafficlight.getPhase(tls_id))
            self._current_phase[tls_id] = cur
            self._steps_on_phase[tls_id] = self.min_green_steps

    def _advance(self, tls_id: str, nxt: int) -> None:
        assert self._current_phase is not None and self._steps_on_phase is not None
        cur = self._current_phase[tls_id]
        if nxt == cur:
            self._steps_on_phase[tls_id] = self._steps_on_phase[tls_id] + 1
        else:
            self._current_phase[tls_id] = nxt
            self._steps_on_phase[tls_id] = 0


@dataclass
class MaxPressureController(_PhaseControllerBase):
    """Фаза с максимальным давлением на въездах/выездах."""

    def compute_actions(self, tls_ids: list[str]) -> dict[str, int]:
        self._ensure_ready()
        assert self._pairs_by_tls is not None
        assert self._current_phase is not None
        assert self._steps_on_phase is not None

        actions: dict[str, int] = {}
        for tls_id in tls_ids:
            pairs_by_phase = self._pairs_by_tls[tls_id]
            cur = self._current_phase[tls_id]
            steps = self._steps_on_phase[tls_id]
            scores: list[float | None] = [
                pressure_for_lane_pairs(p) if p else None for p in pairs_by_phase
            ]
            nxt = _select_best_phase(
                scores,
                current_phase=cur,
                steps_on_phase=steps,
                min_green_steps=self.min_green_steps,
            )
            actions[tls_id] = nxt
            self._advance(tls_id, nxt)
        return actions


@dataclass
class GreedyController(_PhaseControllerBase):
    """Фаза с максимальной суммой очереди и подъезжающих на въездах (RESCO Greedy)."""

    def compute_actions(self, tls_ids: list[str]) -> dict[str, int]:
        self._ensure_ready()
        assert self._pairs_by_tls is not None
        assert self._current_phase is not None
        assert self._steps_on_phase is not None

        actions: dict[str, int] = {}
        for tls_id in tls_ids:
            pairs_by_phase = self._pairs_by_tls[tls_id]
            cur = self._current_phase[tls_id]
            steps = self._steps_on_phase[tls_id]
            scores: list[float | None] = [
                greedy_score_for_phase(p) if p else None for p in pairs_by_phase
            ]
            nxt = _select_best_phase(
                scores,
                current_phase=cur,
                steps_on_phase=steps,
                min_green_steps=self.min_green_steps,
            )
            actions[tls_id] = nxt
            self._advance(tls_id, nxt)
        return actions
