from src.map_engine.converter import MapConverter

def test_osm_to_sumo_invokes_netconvert(monkeypatch):
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