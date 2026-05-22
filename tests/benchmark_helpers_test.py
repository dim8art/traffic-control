from __future__ import annotations

from subprocess import CompletedProcess

from benchmark import _summarize_benchmark_runs, generate_net_variant


def test_generate_net_variant_success_builds_expected_command(monkeypatch):
    """generate_net_variant вызывает netconvert с -s, типом TLS и выходным файлом."""
    calls: dict = {}

    def fake_run(cmd, **_kw):
        calls["cmd"] = cmd
        return CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr("benchmark.subprocess.run", fake_run)

    ok = generate_net_variant("base.net.xml", "actuated", "out.net.xml")
    assert ok is True
    assert calls["cmd"] == [
        "netconvert",
        "-s",
        "base.net.xml",
        "--tls.default-type",
        "actuated",
        "-o",
        "out.net.xml",
    ]


def test_generate_net_variant_handles_netconvert_failure(monkeypatch):
    """Любой сбой subprocess оборачивается в False без необработанного исключения наружу."""
    def boom(*args, **kwargs):  # noqa: ANN001, ANN002
        raise OSError("no netconvert")

    monkeypatch.setattr("benchmark.subprocess.run", boom)
    assert generate_net_variant("base.net.xml", "static", "out.net.xml") is False


def test_summarize_benchmark_runs_computes_mean_std_and_ci():
    runs = [
        {"Static": {"Avg Speed (m/s)": 8.0, "Wait Time (s)": 5.0}},
        {"Static": {"Avg Speed (m/s)": 9.0, "Wait Time (s)": 4.0}},
        {"Static": {"Avg Speed (m/s)": 10.0, "Wait Time (s)": 3.0}},
    ]
    summary = _summarize_benchmark_runs(runs, confidence_level=0.95)
    speed = summary["Static"]["Avg Speed (m/s)"]
    assert speed["mean"] == 9.0
    assert round(speed["std"], 6) == round(1.0, 6)
    assert speed["ci_low"] < speed["mean"] < speed["ci_high"]
    assert speed["n"] == 3.0


def test_summarize_benchmark_runs_single_run_has_zero_std():
    runs = [{"PPO": {"Total Reward": -100.0}}]
    summary = _summarize_benchmark_runs(runs)
    row = summary["PPO"]["Total Reward"]
    assert row["mean"] == -100.0
    assert row["std"] == 0.0
    assert row["ci_low"] == row["ci_high"] == -100.0
