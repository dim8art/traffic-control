from __future__ import annotations

from subprocess import CalledProcessError

from src.map_engine.converter import MapConverter


def test_osm_to_sumo_invokes_netconvert(monkeypatch):
    """Сборка командной строки netconvert: входной OSM, выходная сеть и check=True."""
    called = {}

    def fake_run(command, check):
        called["command"] = command
        called["check"] = check
        return 0

    monkeypatch.setattr("src.map_engine.converter.subprocess.run", fake_run)
    output = MapConverter.osm_to_sumo("input.osm", "mini_map")

    assert output.endswith("data/sumo/mini_map.net.xml")
    assert called["command"][0] == "netconvert"
    assert "--osm-files" in called["command"]
    assert "input.osm" in called["command"]
    assert called["check"] is True


def test_osm_to_sumo_returns_none_when_netconvert_fails(monkeypatch):
    """При ненулевом коде/исключении netconvert возвращается None без падения пайплайна."""
    def fail_run(*args, **kwargs):
        raise CalledProcessError(
            returncode=2,
            cmd=["netconvert"],
            stderr=b"fail",
            output=b"",
        )

    monkeypatch.setattr("src.map_engine.converter.subprocess.run", fail_run)
    assert MapConverter.osm_to_sumo("in.osm", "out_map") is None


def test_osm_to_sumo_fast_mode_flags(monkeypatch):
    """Режим fast=True ослабляет геометрию (--junctions, roundabouts, internal links)."""
    captured: dict = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr("src.map_engine.converter.subprocess.run", fake_run)

    MapConverter.osm_to_sumo("input.osm", "fast_net", fast=True)
    cmd = captured["command"]
    assert "--no-internal-links" in cmd
    idx = cmd.index("--no-internal-links")
    assert cmd[idx + 1] == "true"
    jd = cmd.index("--junctions.join-dist")
    assert cmd[jd + 1] == "18"
    rg = cmd.index("--roundabouts.guess")
    assert cmd[rg + 1] == "false"
