from datetime import datetime

import pytest

from src.simulation.env import REWARD_MODE_ALL, REWARD_MODES

from train import _resolve_tune_restore_dir, build_training_run_name


def test_resolve_tune_restore_dir_trial(tmp_path):
    trial = tmp_path / "PPO_test_00000"
    trial.mkdir()
    (trial / "params.json").write_text("{}", encoding="utf-8")
    ckpt = trial / "checkpoint_000010"
    ckpt.mkdir()
    assert _resolve_tune_restore_dir(str(ckpt)) == str(trial)
    assert _resolve_tune_restore_dir(str(trial)) == str(trial)


def test_resolve_tune_restore_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_tune_restore_dir(str(tmp_path / "nope"))


def test_reward_mode_all_is_six_sequential_modes() -> None:
    assert REWARD_MODE_ALL == "all"
    assert len(REWARD_MODES) == 6


def test_build_training_run_name_includes_timestamp_and_params() -> None:
    name = build_training_run_name(
        "data/sumo/vasilevsky.net.xml",
        0.5,
        3600,
        training_iterations=50,
        reward_mode="pressure_sidewalk",
        enable_pedestrians=True,
        enable_public_transport=False,
        pedestrian_period=None,
        public_transport_period=None,
        ped_reward_weight=1.0,
        num_env_runners=6,
        rollout_fragment_length=50,
        started_at=datetime(2026, 5, 22, 15, 30, 45),
    )
    assert name.startswith("20260522_153045_")
    assert "vasilevsky" in name
    assert "p0p5" in name
    assert "d3600" in name
    assert "rmpressure_sidewalk" in name
    assert "iter50" in name
    assert "pedauto" in name
    assert "pw1" in name
