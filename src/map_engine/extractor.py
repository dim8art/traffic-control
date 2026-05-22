from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
import osmnx as ox
from shapely import box
from shapely.geometry import Polygon

ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.requests_timeout = 300
ox.settings.all_oneway = True

logger = logging.getLogger(__name__)

# Запас по границе bbox для osmium extract (~200 м в широте Петербурга)
_CLIP_PAD_DEG = 0.003


def _osmium_can_clip(path: Path) -> bool:
    n = path.name.lower()
    return bool(
        n.endswith(".pbf")
        or n.endswith(".osm.pbf")
        or n.endswith(".osm")
    )


def load_graph_from_local_osm_extract(
    file_path: str, *, simplify_xml: bool = False
) -> nx.MultiDiGraph:
    """Полная загрузка без пространственного пре-среза (дорого на большом PBF)."""

    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Файл карты не найден: {path}")

    name_lower = path.name.lower()

    if name_lower.endswith(".graphml") or name_lower.endswith(".graphml.gz"):
        return ox.load_graphml(path)

    if name_lower.endswith(".pbf") or name_lower.endswith(".osm.pbf"):
        osmium_bin = shutil.which("osmium")
        if osmium_bin is None:
            raise RuntimeError(
                "Для .pbf нужна утилита osmium-tool (команда `osmium`): "
                "установите пакет или сконвертируйте PBF в .osm вручную."
            )

        fd, tmp_osm = tempfile.mkstemp(suffix=".osm", prefix="osm_extract_")
        os.close(fd)
        logger.info("Полная конвертация PBF → OSM (osmium cat)…")
        try:
            subprocess.run(
                [
                    osmium_bin,
                    "cat",
                    str(path),
                    "-o",
                    tmp_osm,
                    "-f",
                    "osm",
                    "-O",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return ox.graph_from_xml(tmp_osm, simplify=simplify_xml)
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.strip() if exc.stderr else str(exc)
            raise RuntimeError(f"osmium завершился с ошибкой:\n{err}") from exc
        finally:
            try:
                os.unlink(tmp_osm)
            except OSError:
                pass

    return ox.graph_from_xml(path, simplify=simplify_xml)


def load_osm_drive_graph(
    file_path: str,
    *,
    clip_bbox_wsen: tuple[float, float, float, float] | None,
    simplify_xml: bool,
) -> nx.MultiDiGraph:
    """
    clip_bbox_wsen = (west, south, east, north) в градусах.
    Если задан и есть osmium — сначала ``extract`` по bbox → меньше данных для OSMnx/netconvert.
    """

    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Файл карты не найден: {path}")

    name_lower = path.name.lower()

    if name_lower.endswith(".graphml") or name_lower.endswith(".graphml.gz"):
        if clip_bbox_wsen is not None:
            logger.info(
                "GraphML: пре-срез osmium недоступен, граф целиком; затем полигон в OSMnx."
            )
        return ox.load_graphml(path)

    osmium_bin = shutil.which("osmium")
    if (
        clip_bbox_wsen is not None
        and osmium_bin
        and _osmium_can_clip(path)
    ):
        west, south, east, north = clip_bbox_wsen
        fd, tmp_osm = tempfile.mkstemp(suffix=".osm", prefix="osmium_bbox_")
        os.close(fd)
        logger.info(
            "Ускорение: osmium extract по bbox перед OSMnx (west,south,east,north="
            f"{west:.5f},{south:.5f},{east:.5f},{north:.5f})…"
        )
        try:
            subprocess.run(
                [
                    osmium_bin,
                    "extract",
                    "-b",
                    f"{west},{south},{east},{north}",
                    str(path),
                    "-o",
                    tmp_osm,
                    "-f",
                    "osm",
                    "-O",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return ox.graph_from_xml(tmp_osm, simplify=simplify_xml)
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.strip() if exc.stderr else str(exc)
            logger.warning(
                "osmium extract не удался (%s), откат к полной загрузке файла.",
                err[:200],
            )
        finally:
            try:
                os.unlink(tmp_osm)
            except OSError:
                pass
    elif clip_bbox_wsen is not None and not osmium_bin:
        logger.warning(
            "osmium не найден в PATH — пре-срез по bbox пропущен (медленнее на больших PBF)."
        )

    return load_graph_from_local_osm_extract(
        str(path), simplify_xml=simplify_xml
    )


class MapExtractor:
    def __init__(
        self,
        file_path: str | None = None,
        location: tuple[float, float] | None = None,
        dist: int | float | None = None,
        polygon: Polygon | None = None,
        network_type: str = "drive",
        fast_prepare: bool = False,
    ) -> None:
        self.file_path = file_path
        self.location = location
        self.dist = dist
        self.polygon = polygon
        self.network_type = network_type
        self.fast_prepare = fast_prepare

        self.graph = None
        self.signals = None

    def download_and_process(self):
        """Загрузка с диска, опционально с пре-срезом bbox и полигоном OSMnx."""
        try:
            if not self.file_path or not os.path.exists(self.file_path):
                logger.error(
                    "Нужен существующий локальный файл карты (облачная загрузка отключена).",
                )
                return None, None

            simplify_xml = bool(self.fast_prepare)

            target_polygon = self.polygon

            if (
                target_polygon is None
                and self.location is not None
                and self.dist is not None
            ):
                west, south, east, north = ox.utils_geo.bbox_from_point(
                    self.location, dist=float(self.dist)
                )
                target_polygon = box(west, south, east, north)
                logger.info(
                    "Контур обрезки: bbox %.0f м от точки.", float(self.dist)
                )

            clip_bbox = None
            if target_polygon is not None:
                minx, miny, maxx, maxy = target_polygon.bounds
                clip_bbox = (
                    minx - _CLIP_PAD_DEG,
                    miny - _CLIP_PAD_DEG,
                    maxx + _CLIP_PAD_DEG,
                    maxy + _CLIP_PAD_DEG,
                )

            logger.info(f"Чтение графа: {self.file_path}")
            full_graph = load_osm_drive_graph(
                self.file_path,
                clip_bbox_wsen=clip_bbox,
                simplify_xml=simplify_xml,
            )

            if target_polygon is not None:
                logger.info("Точная обрезка по полигону (OSMnx)…")
                self.graph = ox.truncate.truncate_graph_polygon(
                    full_graph,
                    target_polygon,
                    truncate_by_edge=True,
                )
            else:
                self.graph = full_graph

        except Exception as e:
            logger.error("Ошибка при обработке карты: %s", e)
            return None, None

        logger.info("Выбор узлов светофоров…")
        nodes, _ = ox.graph_to_gdfs(self.graph)
        
        if 'highway' in nodes.columns:
            # 1. Обычные узлы-светофоры.
            standard_signals = nodes[nodes['highway'] == 'traffic_signals']
            
            # 2. Переходы, помеченные как traffic_signals (иногда highway=crossing).
            crossing_signals = nodes[nodes['highway'].isin(['crossing']) & 
                                    (nodes.get('crossing') == 'traffic_signals')]
            
            import pandas as pd
            self.signals = pd.concat([standard_signals, crossing_signals]).drop_duplicates()
        else:
            self.signals = []

        logger.info("Узлов в графе: %s", len(self.graph.nodes))
        return self.graph, self.signals

    def get_adjacency_list(self):
        """
        Карта соседей для распределённых агентов: ключ — id сигнала, значение — id смежных узлов.
        """
        adj_map = {}
        if self.signals is None:
            return adj_map
            
        signal_ids = self.signals.index.tolist()
        for node_id in signal_ids:
            neighbors = list(self.graph.neighbors(node_id))
            adj_map[node_id] = neighbors
            
        return adj_map

    def save_osm_xml(self, filename):
        """Сохранение графа в нативном OSM XML для совместимости с SUMO."""
        output_dir = "data/network"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filepath = os.path.join(output_dir, f"{filename}.osm")
        ox.save_graph_xml(self.graph, filepath)
        logger.info("Сохранён OSM XML: %s", filepath)
        return filepath

    def visualize_with_signals(self, save_path=None):
        if self.graph is None or len(self.graph.nodes) == 0:
            logger.error("Граф пуст.")
            return

        node_colors = []
        node_sizes = []
        
        signal_ids = set(self.signals.index) if hasattr(self.signals, 'index') else set()

        for node in self.graph.nodes():
            if node in signal_ids:
                node_colors.append('#FF4500')  # оранжево-красный для светофоров
                node_sizes.append(50)
            else:
                node_colors.append('#666666')  # серый для обычных узлов
                node_sizes.append(15)


        if not node_sizes:
            node_sizes = 15  # стандартный размер, если списки пусты
            node_colors = '#666666'

        try:
            import matplotlib.pyplot as plt
            
            fig, ax = ox.plot_graph(
                self.graph,
                node_color=node_colors,
                node_size=node_sizes,
                node_alpha=0.8,
                edge_color='#CCCCCC',
                edge_linewidth=0.8,
                bgcolor='#FFFFFF',
                show=False,
                close=False
            )
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, format='svg' if save_path.endswith('.svg') else 'png')
                plt.close(fig)
                logger.info("Превью сохранено: %s", save_path)

        except Exception as e:
            logger.error("Ошибка при сохранении превью графа: %s", e)