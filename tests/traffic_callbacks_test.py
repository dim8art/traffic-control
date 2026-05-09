from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from src.simulation.env import TrafficCallbacks


@dataclass
class _EpisodeStub:
    """Минимальная заглушка episode для RLlib callbacks (info и user_data/custom_metrics)."""
    user_data: dict = field(default_factory=dict)
    custom_metrics: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def last_info_for(self, key: str) -> dict | None:
        del key  # только __common__
        return self.info if self.info else None


def test_traffic_callbacks_accumulate_waiting_and_speed() -> None:
    """on_episode_step копит wait/speed из __common__, on_episode_end пишет средние в custom_metrics."""
    cb = TrafficCallbacks()
    ep = _EpisodeStub(
        info={
            "system_mean_waiting_time": 4.0,
            "system_avg_speed": 8.0,
        }
    )

    dummy = MagicMock()
    kwargs = {}

    cb.on_episode_step(
        worker=dummy,
        base_env=dummy,
        policies={},
        episode=ep,
        env_index=0,
        **kwargs,
    )
    cb.on_episode_step(
        worker=dummy,
        base_env=dummy,
        policies={},
        episode=ep,
        env_index=0,
        **kwargs,
    )

    assert ep.user_data["waiting_times"] == [4.0, 4.0]
    assert ep.user_data["avg_speeds"] == [8.0, 8.0]

    cb.on_episode_end(
        worker=dummy,
        base_env=dummy,
        policies={},
        episode=ep,
        env_index=0,
        **kwargs,
    )

    assert ep.custom_metrics["system_mean_waiting_time"] == 4.0
    assert ep.custom_metrics["system_avg_speed"] == 8.0
