"""
Интерактивное меню: режимы и параметры запуска.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import questionary
from questionary import Choice


def _abspath(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    raw = raw.strip()
    return os.path.abspath(os.path.expanduser(raw)) if raw else None


def _prompt_text(title: str, default: Optional[str] = None) -> Optional[str]:
    return questionary.text(title, default=default or "").ask()


def _prompt_int(title: str, default: str) -> int:
    raw = questionary.text(title, default=default).ask()
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        questionary.print(f"Не число, подставлено значение по умолчанию: {default}")
        return int(default)


def _prompt_float(title: str, default: str) -> float:
    raw = questionary.text(title, default=default).ask()
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        questionary.print(f"Не число, подставлено значение по умолчанию: {default}")
        return float(default)


def _prompt_existing_path(title: str) -> Optional[str]:
    raw = questionary.path(title).ask()
    path = _abspath(raw)
    if path is None:
        return None
    return path if os.path.exists(path) else None


def _region_preset_choices() -> list[Choice[str]]:
    from prepare_map import AREA_LABELS, NAMED_AREAS

    pairs = [(AREA_LABELS.get(k, k), k) for k in NAMED_AREAS]
    pairs.sort(key=lambda x: x[0].casefold())
    return [Choice(title=t, value=v) for t, v in pairs]


def _loop_prepare() -> None:
    from prepare_map import AREA_LABELS, KOLOMNA_POLY, main as prepare_main

    map_path = _prompt_existing_path("Файл карты (OSM, PBF или GraphML)")
    if not map_path:
        questionary.print("Файл не найден.")
        return

    mode = questionary.select(
        "Фрагмент сети",
        choices=[
            "Центр и радиус",
            "Готовый регион",
            "Свой полигон",
            questionary.Separator(),
            "Назад",
        ],
    ).ask()

    if mode is None or mode == "Назад":
        return

    bbox: Optional[Tuple[float, float, float]] = None
    polygon_coords: Optional[str] = None
    area: Optional[str] = None
    name_default = "network"

    if mode == "Центр и радиус":
        lt = (_prompt_text("Широта", "59.932") or "").strip()
        lg = (_prompt_text("Долгота", "30.30") or "").strip()
        try:
            lat_v, lon_v = float(lt), float(lg)
        except ValueError:
            questionary.print("Некорректные координаты.")
            return
        r_v = float(_prompt_int("Радиус, м", "500"))
        bbox = (lat_v, lon_v, r_v)
        name_default = "bbox_fragment"
    elif mode == "Готовый регион":
        picked = questionary.select(
            "Регион (Санкт-Петербург, пресеты)",
            choices=_region_preset_choices(),
        ).ask()
        if picked is None:
            return
        area = str(picked)
        name_default = area
    else:
        poly = questionary.text(
            "Полигон: lon,lat через точку с запятой",
            default=KOLOMNA_POLY,
        ).ask()
        polygon_coords = (poly or "").strip() or KOLOMNA_POLY
        name_default = "polygon_fragment"

    name = _prompt_text(
        "Имя набора файлов → data/network/<имя>.osm и data/sumo/<имя>.net.xml",
        name_default,
    )
    if name is None:
        return
    name = name.strip()
    if not name:
        questionary.print("Отмена.")
        return

    if area:
        questionary.print(
            f"Пресет: {AREA_LABELS.get(area, area)} → примите имя «{name}» или смените его."
        )

    save_preview = questionary.confirm(
        "Сохранить изображение превью сети (SVG)?", default=False
    ).ask()
    if save_preview is None:
        return

    questionary.print("Подготовка…")
    prepare_main(
        map_path,
        name,
        bbox=bbox,
        polygon_coords=polygon_coords,
        area=area,
        preview=save_preview,
    )


def _loop_train() -> None:
    from train import run_train

    net_xml = _prompt_existing_path("Сеть SUMO (.net.xml)")
    if not net_xml:
        return
    period = _prompt_float("Интервал выпуска ТС (меньше = плотнее)", "0.5")
    duration = _prompt_int("Длительность, с", "3600")
    questionary.print("Обучение…")
    run_train(net_xml, period, duration)


def _loop_sumo_freerun() -> None:
    from demo import run_sumo_freerun

    net_xml = _prompt_existing_path("Сеть SUMO (.net.xml)")
    if not net_xml:
        return
    period = _prompt_float("Интервал выпуска ТС", "0.4")
    duration = _prompt_int("Длительность, с", "3600")
    gui = questionary.confirm("Окно SUMO?", default=True).ask()
    if gui is None:
        return
    questionary.print("Симуляция…")
    run_sumo_freerun(net_xml, duration, period, gui)


def _loop_demo() -> None:
    from demo import run_inference

    ckpt = _prompt_existing_path("Каталог checkpoint")
    if not ckpt or not os.path.isdir(ckpt):
        questionary.print("Укажите существующий каталог.")
        return
    net_xml = _prompt_existing_path("Сеть SUMO (.net.xml)")
    if not net_xml:
        return
    period = _prompt_float("Интервал выпуска ТС", "0.4")
    duration = _prompt_int("Длительность, с", "3600")
    gui = questionary.confirm("Окно SUMO?", default=True).ask()
    if gui is None:
        return
    questionary.print("Запуск…")
    run_inference(ckpt, net_xml, duration, period, gui)


def _loop_benchmark() -> None:
    from benchmark import run_comprehensive_benchmark

    ckpt = _prompt_existing_path("Каталог checkpoint")
    if not ckpt or not os.path.isdir(ckpt):
        questionary.print("Укажите существующий каталог.")
        return
    net_xml = _prompt_existing_path("Сеть SUMO (.net.xml)")
    if not net_xml:
        return
    period = _prompt_float("Интервал выпуска ТС", "0.4")
    duration = _prompt_int("Длительность, с", "3600")
    questionary.print("Сравнение…")
    run_comprehensive_benchmark(ckpt, net_xml, duration, period)


def run_tui() -> None:
    while True:
        choice = questionary.select(
            "Traffic control",
            choices=[
                "Подготовка сети",
                "Обучение",
                "Запуск в SUMO (модель)",
                "SUMO без модели",
                "Сравнение режимов",
                questionary.Separator(),
                "Выход",
            ],
        ).ask()

        if choice is None or choice == "Выход":
            return

        try:
            if choice == "Подготовка сети":
                _loop_prepare()
            elif choice == "Обучение":
                _loop_train()
            elif choice == "Запуск в SUMO (модель)":
                _loop_demo()
            elif choice == "SUMO без модели":
                _loop_sumo_freerun()
            elif choice == "Сравнение режимов":
                _loop_benchmark()
        except KeyboardInterrupt:
            questionary.print("\nОтменено.")
            continue


def main() -> None:
    try:
        run_tui()
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
