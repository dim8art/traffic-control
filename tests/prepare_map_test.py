from __future__ import annotations

from pathlib import Path

import argparse
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import Polygon

import prepare_map
from prepare_map import NAMED_AREAS, register_prepare_arguments, _polygon_from_string


def test_polygon_from_string_rectangle() -> None:
    """Корректная строка lon,lat через «;» даёт замкнутый валидный полигон Shapely."""
    coords = "30.0,59.9;31.0,59.9;31.0,60.0;30.0,60.0;30.0,59.9"
    poly = _polygon_from_string(coords)
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    assert poly.area > 0


def test_polygon_from_string_invalid_raises() -> None:
    """При невозможности разобрать координаты float() пробрасывает ValueError."""
    with pytest.raises(ValueError):
        _polygon_from_string("not_a_lon,lat;2,3")


@pytest.mark.parametrize("area", sorted(NAMED_AREAS))
def test_named_area_polygons_parse_and_valid(area: str) -> None:
    """Каждый пресет --area задаёт строку координат, пригодную для обрезки карты."""
    poly = _polygon_from_string(NAMED_AREAS[area])
    assert poly.is_valid
    assert poly.area > 0


def test_prepare_cli_mutually_exclusive_zones(tmp_path: Path, monkeypatch) -> None:
    """CLI prepare: одновременно --bbox и --area должны давать ошибку argparse (выход≠0)."""
    monkeypatch.chdir(tmp_path)
    p = argparse.ArgumentParser()
    register_prepare_arguments(p)
    with pytest.raises(SystemExit):
        p.parse_args(
            [
                "--map",
                "x.osm",
                "--name",
                "t",
                "--bbox",
                "59.9",
                "30.3",
                "500",
                "--area",
                "kolomna",
            ]
        )


def test_prepare_main_happy_path_uses_converter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """При успешной выгрузке графа main вызывает конвертер с правильными именами и путём к OSM."""
    monkeypatch.chdir(tmp_path)
    map_file = tmp_path / "fake.osm"
    map_file.write_text("<osm></osm>", encoding="utf-8")

    g = nx.MultiDiGraph()
    g.add_node(1, x=30.3, y=59.9)

    def fake_dl(self):  # noqa: ANN001
        return g, pd.DataFrame()

    def fake_save_osm(self, name: str) -> Path:  # noqa: ANN001
        out_dir = tmp_path / "data" / "network"
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{name}.osm"
        p.touch()
        return p

    convert_calls: dict[str, object] = {}

    def fake_convert(input_path: str, output_filename: str, *, fast: bool = False):  # noqa: ANN201
        convert_calls["input_path"] = input_path
        convert_calls["output_filename"] = output_filename
        convert_calls["fast"] = fast
        return "/dev/null/mock.net.xml"

    monkeypatch.setattr(prepare_map.MapExtractor, "download_and_process", fake_dl)
    monkeypatch.setattr(prepare_map.MapExtractor, "save_osm_xml", fake_save_osm)
    monkeypatch.setattr(prepare_map.MapConverter, "osm_to_sumo", staticmethod(fake_convert))

    prepare_map.main(str(map_file), "testnet", bbox=(59.932, 30.349, 400.0), fast=False)

    assert convert_calls["output_filename"] == "testnet"
    assert convert_calls["fast"] is False
    xml_path = tmp_path / "data" / "network" / "testnet.osm"
    assert Path(convert_calls["input_path"]).resolve() == xml_path.resolve()


def test_prepare_main_missing_map_file(tmp_path: Path, monkeypatch, caplog) -> None:
    """Если файл карты не существует, обработка прерывается и пишется ERROR в лог."""
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("ERROR"):
        prepare_map.main(
            "/no/such/map.osm",
            "x",
            bbox=(59.0, 30.0, 100.0),
        )
    assert ("не найден" in caplog.text) or ("not found" in caplog.text.lower())
