import os

import pytest

from train import _resolve_tune_restore_dir


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
