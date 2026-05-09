import os
from subprocess import CalledProcessError

os.environ.setdefault("SUMO_HOME", "/tmp")

from src.simulation.runner import SumoRunner


def test_generate_random_traffic_fallback_writes_empty_routes(tmp_path, monkeypatch):
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


def test_create_config_writes_sumocfg(tmp_path):
    net_file = tmp_path / "demo.net.xml"
    net_file.write_text("<net></net>", encoding="utf-8")

    runner = SumoRunner(str(net_file), output_dir=str(tmp_path / "sumo"), unique_id="cfg")
    runner.create_config()

    assert os.path.exists(runner.cfg_file)
    content = open(runner.cfg_file, "r", encoding="utf-8").read()
    assert "<configuration>" in content
    assert str(net_file.name) not in content  # stored as copied worker-local net file