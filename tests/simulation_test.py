import os
from subprocess import CalledProcessError, CompletedProcess

os.environ.setdefault("SUMO_HOME", "/tmp")

from src.simulation.env import traffic_env_config, traffic_observation_dim
from src.simulation.runner import SumoRunner


def test_generate_random_traffic_fallback_writes_empty_routes(tmp_path, monkeypatch):
    """При падении randomTrips создаётся пустой routes.rou.xml, чтобы SUMO мог стартовать деградированно."""
    net_file = tmp_path / "demo.net.xml"
    net_file.write_text("<net></net>", encoding="utf-8")
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "randomTrips.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setenv("SUMO_HOME", str(tmp_path))

    def fake_run(*args, **kwargs):
        raise CalledProcessError(returncode=1, cmd="randomTrips.py", stderr="failed")

    monkeypatch.setattr("src.simulation.runner.subprocess.run", fake_run)

    runner = SumoRunner(str(net_file), output_dir=str(tmp_path / "sumo"), unique_id="test")
    runner.generate_random_traffic(period=1.0, duration=10)

    assert os.path.exists(runner.rou_file)
    content = open(runner.rou_file, "r", encoding="utf-8").read()
    assert content.strip() == "<routes></routes>"


def test_generate_random_traffic_success_invokes_random_trips(tmp_path, monkeypatch):
    """Успешный вызов проксирует интервал периода -p и конец симуляции -e в команду инструментов SUMO."""
    net_file = tmp_path / "demo.net.xml"
    net_file.write_text("<net></net>", encoding="utf-8")
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "randomTrips.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setenv("SUMO_HOME", str(tmp_path))

    captured = {}

    def fake_run(cmd, check=True, capture_output=False, text=False):
        captured["cmd"] = cmd
        return CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.simulation.runner.subprocess.run", fake_run)

    runner = SumoRunner(
        str(net_file), output_dir=str(tmp_path / "sumo"), unique_id="ok"
    )
    runner.generate_random_traffic(period=0.75, duration=3600)

    cmd = captured["cmd"]
    assert "randomTrips.py" in cmd[1] or cmd[1].endswith("randomTrips.py")
    assert "-p" in cmd
    idx = cmd.index("-p")
    assert cmd[idx + 1] == "0.75"
    assert "-e" in cmd
    idx_e = cmd.index("-e")
    assert cmd[idx_e + 1] == "3600"
    idx_pf = cmd.index("--prefix")
    assert cmd[idx_pf + 1] == "veh"


def test_create_config_writes_sumocfg(tmp_path):
    """Создаётся sim.sumocfg с абсолютными путями к сетке и роутерам во временной директории воркера."""
    net_file = tmp_path / "demo.net.xml"
    net_file.write_text("<net></net>", encoding="utf-8")

    runner = SumoRunner(str(net_file), output_dir=str(tmp_path / "sumo"), unique_id="cfg")
    runner.create_config()

    assert os.path.exists(runner.cfg_file)
    content = open(runner.cfg_file, "r", encoding="utf-8").read()
    assert "<configuration>" in content
    assert str(net_file.name) not in content  # stored as copied worker-local net file
    assert '<end value="3600"/>' in content
    assert "ignore-route-errors" in content


def test_traffic_observation_dim_matches_env_flags() -> None:
    assert traffic_observation_dim() == 6
    assert traffic_observation_dim(enable_pedestrians=True) == 8
    assert traffic_observation_dim(enable_public_transport=True) == 8
    assert traffic_observation_dim(
        enable_pedestrians=True, enable_public_transport=True
    ) == 10


def test_traffic_env_config_omits_none_periods():
    cfg = traffic_env_config(
        "/tmp/map.net.xml",
        traffic_period=0.3,
        duration=120,
        enable_pedestrians=True,
    )
    assert cfg["enable_pedestrians"] is True
    assert "pedestrian_period" not in cfg


def test_generate_random_traffic_with_pedestrians_calls_random_trips_twice(
    tmp_path, monkeypatch
):
    net_file = tmp_path / "demo.net.xml"
    net_file.write_text("<net></net>", encoding="utf-8")
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "randomTrips.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setenv("SUMO_HOME", str(tmp_path))

    captured: list[list[str]] = []

    def fake_run(cmd, check=True, capture_output=False, text=False):
        captured.append(list(cmd))
        return CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.simulation.runner.subprocess.run", fake_run)

    runner = SumoRunner(str(net_file), output_dir=str(tmp_path / "sumo"), unique_id="ped")
    runner.generate_random_traffic(
        period=0.5, duration=1800, enable_pedestrians=True
    )

    assert len(captured) == 2
    assert "--validate" in captured[0]
    assert captured[0][captured[0].index("--prefix") + 1] == "veh"
    assert "--validate" not in captured[1]
    assert "--pedestrians" in captured[1]
    assert captured[1][captured[1].index("--prefix") + 1] == "ped"
    assert "-t" not in captured[1]
    assert runner._include_pedestrians is True