"""Юнит-тесты чистой логики saliency (без Ray/чекпоинта)."""

from __future__ import annotations

import numpy as np

from src.rl.obs_salience import (
    format_top_saliency,
    log_saliency_tensorboard_scalar_groups,
    mean_abs_saliency_across_tls,
    observation_feature_labels,
)


def test_observation_feature_labels_six_known_names() -> None:
    """Размер 6 даёт имена, согласованные с MultiAgentTrafficEnv._get_local_obs."""
    names = observation_feature_labels(6)
    assert len(names) == 6
    assert names[0] == "local_waiting"


def test_observation_feature_labels_extended() -> None:
    """Лишние измерения без флагов получают технические имена f{i}."""
    names = observation_feature_labels(8)
    assert names[6] == "f6"


def test_observation_feature_labels_with_ped_and_pt() -> None:
    names = observation_feature_labels(
        10,
        enable_pedestrians=True,
        enable_public_transport=True,
    )
    assert len(names) == 10
    assert names[6] == "local_ped_count"
    assert names[7] == "neighbor_ped_count_w"
    assert names[8] == "local_halting_buses"
    assert names[9] == "neighbor_halting_buses_w"


def test_mean_abs_saliency_across_tls() -> None:
    """Среднее |g| по двум наблюденным TLS даёт среднюю строку нужной размерности."""
    a = np.array([10.0, 0.0, 6.0, 4.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([10.0, 0.0, 6.0, 0.0, 12.0, 0.0], dtype=np.float32)
    m = mean_abs_saliency_across_tls([a, b])
    np.testing.assert_allclose(m, [10.0, 0.0, 6.0, 2.0, 6.0, 0.0])


class _DummyTBWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))


def test_log_saliency_tensorboard_writes_shares_sum_to_one() -> None:
    """Доля share_sum ко всём компонентам в сумме ≈ 1 для одноступенчатый лога."""
    w = _DummyTBWriter()
    mean_abs = np.array([0.3, 0.7], dtype=np.float32)
    log_saliency_tensorboard_scalar_groups(w, 42, mean_abs, prefix="t")
    share_vals = [
        val for tag, val, st in w.scalars if tag.startswith("t/share_sum/") and st == 42
    ]
    assert abs(sum(share_vals) - 1.0) < 1e-5


def test_format_top_saliency_orders_descending() -> None:
    """Печать сортирует по величине |grad| по убыванию."""
    grads = np.array([0.01, 10.0, 2.5, 0.0, -3.0, 0.05], dtype=np.float32)
    s = format_top_saliency(grads, top_k=3, abs_values=True)
    assert "local_speed=" in s
    assert "neighbor_speed_w=" in s
    assert s.count(", ") == 2
