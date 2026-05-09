import argparse
import os
import sys

import osmnx as ox
from shapely.geometry import Polygon

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.map_engine.extractor import MapExtractor
from src.map_engine.converter import MapConverter

KRESTOVSKY_POLY = (
    "30.200,59.970;30.210,59.983;30.245,59.985;30.275,59.980;30.280,59.975;30.275,59.967;30.250,59.962;30.220,59.962;30.200,59.970"
)
KOLOMNA_POLY = (
    "30.288,59.935;30.305,59.932;30.308,59.925;30.295,59.918;30.285,59.919;30.275,59.928;30.288,59.935"
)

VASILEVSKY_POLY = (
    "30.150,59.925;30.210,59.910;30.265,59.925;30.310,59.945;30.280,59.955;"
    "30.245,59.965;30.180,59.955;30.150,59.925"
)


KANONERSKY_POLY = "30.185,59.885;30.235,59.885;30.235,59.915;30.185,59.915;30.185,59.885"


PETROGRADSKY_POLY = "30.195,59.940;30.345,59.940;30.345,59.985;30.195,59.985;30.195,59.940"


CENTER_ADMIRALTY_POLY = "30.250,59.895;30.400,59.895;30.400,59.950;30.250,59.950;30.250,59.895"


MOSKOVSKY_POLY = "30.270,59.760;30.410,59.760;30.410,59.890;30.270,59.890;30.270,59.760"


PRIMORSKY_POLY = "30.000,59.970;30.320,59.970;30.320,60.060;30.000,60.060;30.000,59.970"


FRUNZENSKY_POLY = "30.350,59.820;30.450,59.820;30.450,59.910;30.350,59.910;30.350,59.820"


VYBORGSKY_POLY = "30.290,59.960;30.380,59.960;30.380,60.080;30.290,60.080;30.290,59.960"


NEVSKY_DISTRICT_POLY = "30.380,59.820;30.520,59.820;30.520,59.930;30.380,59.930;30.380,59.820"

KALININSKY_POLY = "30.350,59.950;30.460,59.950;30.460,60.060;30.350,60.060;30.350,59.950"

NAMED_AREAS = {
    "kolomna": KOLOMNA_POLY,
    "krestovsky": KRESTOVSKY_POLY,
    "vasilevsky": VASILEVSKY_POLY,
    "kanonersky": KANONERSKY_POLY,
    "petrogradsky": PETROGRADSKY_POLY,
    "center_admiralty": CENTER_ADMIRALTY_POLY,
    "moskovsky": MOSKOVSKY_POLY,
    "primorsky": PRIMORSKY_POLY,
    "frunzensky": FRUNZENSKY_POLY,
    "vyborgsky": VYBORGSKY_POLY,
    "nevsky_district": NEVSKY_DISTRICT_POLY,
    "kalininsky": KALININSKY_POLY,
}

# Подписи для интерфейса (TUI и сообщения CLI)
AREA_LABELS = {
    "kolomna": "Коломна",
    "krestovsky": "Крестовский остров",
    "vasilevsky": "Васильевский остров",
    "kanonersky": "Канонерский остров",
    "petrogradsky": "Петроградская сторона",
    "center_admiralty": "Центр (Адмиралтейка, компактно)",
    "moskovsky": "Московский район · крупно",
    "primorsky": "Приморский район · крупно",
    "frunzensky": "Фрунзенский район",
    "vyborgsky": "Выборгский район",
    "nevsky_district": "Невский район",
    "kalininsky": "Калининский район · север",
}

ox.settings.use_cache = True


def _polygon_from_string(polygon_coords: str) -> Polygon:
    points = [tuple(map(float, p.split(","))) for p in polygon_coords.split(";")]
    return Polygon(points)


def main(
    map_path: str,
    name: str,
    *,
    bbox: tuple[float, float, float] | None = None,
    polygon_coords: str | None = None,
    area: str | None = None,
    preview: bool = False,
    fast: bool = False,
) -> None:
    """Build SUMO network from a **local** OSM/PBF/graphml + region filter."""

    osm_xml_path = f"data/network/{name}.osm"

    os.makedirs("data/network", exist_ok=True)
    os.makedirs("data/sumo", exist_ok=True)

    if not os.path.isfile(os.path.expanduser(map_path)):
        print(f"Error: map file not found: {map_path}")
        return

    target_polygon: Polygon | None = None
    lat: float | None = None
    lon: float | None = None
    radius: float | None = None

    mode_count = sum(x is not None for x in (bbox, polygon_coords, area))
    if mode_count != 1:
        print("Internal error: exactly one of bbox, polygon, or area must be set.")
        return

    if area is not None:
        if area not in NAMED_AREAS:
            print(f"Unknown --area {area!r}. Choose from: {sorted(NAMED_AREAS)}")
            return
        try:
            target_polygon = _polygon_from_string(NAMED_AREAS[area])
        except ValueError as e:
            print(f"Error parsing built-in polygon for area {area!r}: {e}")
            return
        print(f"Область пресета: {AREA_LABELS.get(area, area)} ({area})")
    elif polygon_coords is not None:
        try:
            target_polygon = _polygon_from_string(polygon_coords)
            print(f"Using custom polygon with {len(target_polygon.exterior.coords)} vertices")
        except Exception as e:
            print(f"Error parsing polygon: {e}")
            return
    else:
        assert bbox is not None
        lat_f, lon_f, r_f = bbox
        lat, lon = float(lat_f), float(lon_f)
        radius = float(r_f)
        print(f"Truncation: center ({lat}, {lon}), radius {radius} m")

    print(f"--- Processing: {name} (source map: {map_path}) ---")

    extractor = MapExtractor(
        file_path=os.path.expanduser(map_path),
        location=(lat, lon) if lat is not None else None,
        dist=int(radius) if radius is not None else None,
        polygon=target_polygon if bbox is None else None,
        fast_prepare=fast,
    )

    graph, signals = extractor.download_and_process()

    if graph is None or len(graph.nodes) == 0:
        print(
            "Error: empty graph after load/truncate. "
            "Check that the bbox/polygon overlaps your local extract."
        )
        return

    if preview:
        extractor.visualize_with_signals(save_path=f"data/network/{name}_preview.svg")

    extractor.save_osm_xml(name)
    print(f"OSM XML saved. Found {len(signals)} traffic signals (TLS).")

    print("Converting to SUMO format...")
    converter = MapConverter()
    success = converter.osm_to_sumo(
        input_path=osm_xml_path, output_filename=name, fast=fast
    )

    if success:
        print(f"Success: Network '{name}' is ready at data/sumo/{name}.net.xml")
    else:
        print("Error: SUMO conversion failed.")


def register_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared CLI for ``prepare`` (used by ``prepare_map.py`` and ``main.py``)."""

    parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="Локальный файл: .osm, .pbf (нужен osmium), .graphml",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Имя выходных файлов (например kolomna → data/sumo/kolomna.net.xml)",
    )
    zone = parser.add_mutually_exclusive_group(required=True)
    zone.add_argument(
        "--bbox",
        nargs=3,
        metavar=("LAT", "LON", "RADIUS_M"),
        type=float,
        help="Обрезка по bbox: центр LAT, LON и полуразмер RADIUS_M (метры, как в OSMnx bbox_from_point)",
    )
    zone.add_argument(
        "--polygon",
        type=str,
        help="Полигон: lon1,lat1;lon2,lat2;...",
    )
    zone.add_argument(
        "--area",
        type=str,
        choices=sorted(NAMED_AREAS.keys()),
        help=f"Именованный пресет (доступно: {', '.join(sorted(NAMED_AREAS.keys()))})",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Ускорение: упрощения OSMnx и netconvert",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="SVG-превью (дольше на крупной сети)",
    )


def _build_prepare_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    register_prepare_arguments(parser)
    return parser


def run_prepare_from_args(args: argparse.Namespace) -> None:
    raw_bbox = getattr(args, "bbox", None)
    bbox = tuple(raw_bbox) if raw_bbox is not None else None
    main(
        args.map,
        args.name,
        bbox=bbox,
        polygon_coords=args.polygon,
        area=args.area,
        preview=getattr(args, "preview", False),
        fast=getattr(args, "fast", False),
    )


if __name__ == "__main__":
    parser = _build_prepare_parser(
        "Локальный OSM/PBF/graphml → обрезка → SUMO .net.xml (без загрузки из сети)"
    )
    run_prepare_from_args(parser.parse_args())
