"""
Интерактивное меню: режимы и параметры запуска.
"""

from __future__ import annotations

import os
import sys

import questionary
from questionary import Choice


def _abspath(raw: str | None) -> str | None:
    if raw is None:
        return None
    raw = raw.strip()
    return os.path.abspath(os.path.expanduser(raw)) if raw else None


def _prompt_text(title: str, default: str | None = None) -> str | None:
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


def _prompt_simulation_traffic_extras() -> dict[str, object] | None:
    """
    Пешеходы / ОТ — те же опции, что --with-pedestrians и --with-public-transport в CLI.
    Возвращает None, если пользователь прервал подтверждение.
    """
    ped = questionary.confirm(
        "Добавить пешеходов (нужна пешеходная сеть в .net.xml)?",
        default=False,
    ).ask()
    if ped is None:
        return None
    pt = questionary.confirm(
        "Добавить общественный транспорт (автобусы, vclass=bus)?",
        default=False,
    ).ask()
    if pt is None:
        return None

    ped_period: float | None = None
    if ped:
        custom = questionary.confirm(
            "Задать интервал пешеходов вручную (иначе от автоматического правила)?",
            default=False,
        ).ask()
        if custom is None:
            return None
        if custom:
            ped_period = _prompt_float("Интервал между пешеходами, с", "2.0")

    pt_period: float | None = None
    if pt:
        custom = questionary.confirm(
            "Задать интервал автобусов вручную?",
            default=False,
        ).ask()
        if custom is None:
            return None
        if custom:
            pt_period = _prompt_float("Интервал рейсов ОТ, с", "3.0")

    return {
        "enable_pedestrians": bool(ped),
        "enable_public_transport": bool(pt),
        "pedestrian_period": ped_period,
        "public_transport_period": pt_period,
    }


def _prompt_reward_extras() -> dict[str, object] | None:
    """reward_mode и опционально ped_reward_weight для *_sidewalk."""
    from src.simulation.env import REWARD_MODES

    choices = [Choice(title=rm, value=rm) for rm in REWARD_MODES]
    rm = questionary.select(
        "Режим награды",
        choices=choices,
        default="pressure",
    ).ask()
    if rm is None:
        return None
    ped_w = None
    if "sidewalk" in str(rm):
        custom = questionary.confirm(
            "Задать вес пешеходов вручную (иначе по умолчанию 1.0)?",
            default=False,
        ).ask()
        if custom is None:
            return None
        if custom:
            ped_w = _prompt_float("Вес пешеходов", "1.0")
    return {"reward_mode": rm, "ped_reward_weight": ped_w}


def _prompt_demo_saliency() -> dict[str, object] | None:
    """Опции saliency для run_inference (консоль и при желании TensorBoard)."""
    use = questionary.confirm(
        "Saliency: печать |∂log π/∂obs| по наблюдению в консоль?",
        default=False,
    ).ask()
    if use is None:
        return None
    out: dict[str, object] = {
        "explain_obs_every": None,
        "explain_tls_limit": 2,
        "tensorboard_dir": None,
        "tensorboard_saliency_every": 1,
    }
    if use:
        out["explain_obs_every"] = _prompt_int("Печатать saliency каждые N шагов RL", "1")
        out["explain_tls_limit"] = _prompt_int(
            "Не больше скольких светофоров выводить за шаг", "2"
        )
        tb = questionary.confirm("Дополнительно: писать saliency в TensorBoard?", default=False).ask()
        if tb is None:
            return None
        if tb:
            raw_dir = _prompt_text("Каталог TensorBoard", "runs/saliency_demo")
            out["tensorboard_dir"] = (raw_dir or "").strip() or "runs/saliency_demo"
            out["tensorboard_saliency_every"] = _prompt_int(
                "Запись в TensorBoard каждые N шагов RL", "1"
            )
    return out


def _prompt_existing_path(title: str) -> str | None:
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

    bbox: tuple[float, float, float] | None = None
    polygon_coords: str | None = None
    area: str | None = None
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
    traffic_extras = _prompt_simulation_traffic_extras()
    if traffic_extras is None:
        return
    reward_extras = _prompt_reward_extras()
    if reward_extras is None:
        return
    questionary.print("Обучение…")
    run_train(net_xml, period, duration, **traffic_extras, **reward_extras)


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
    traffic_extras = _prompt_simulation_traffic_extras()
    if traffic_extras is None:
        return
    reward_extras = _prompt_reward_extras()
    if reward_extras is None:
        return
    questionary.print("Симуляция…")
    run_sumo_freerun(
        net_xml,
        duration,
        period,
        gui,
        **traffic_extras,
        **reward_extras,
    )


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
    traffic_extras = _prompt_simulation_traffic_extras()
    if traffic_extras is None:
        return
    reward_extras = _prompt_reward_extras()
    if reward_extras is None:
        return
    saliency_opts = _prompt_demo_saliency()
    if saliency_opts is None:
        return
    questionary.print("Запуск…")
    run_inference(
        ckpt,
        net_xml,
        duration,
        period,
        gui,
        **traffic_extras,
        **reward_extras,
        **saliency_opts,
    )


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
    traffic_extras = _prompt_simulation_traffic_extras()
    if traffic_extras is None:
        return
    reward_extras = _prompt_reward_extras()
    if reward_extras is None:
        return
    questionary.print("Сравнение…")
    run_comprehensive_benchmark(
        ckpt,
        net_xml,
        duration,
        period,
        **traffic_extras,
        **reward_extras,
    )


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
            match choice:
                case "Подготовка сети":
                    _loop_prepare()
                case "Обучение":
                    _loop_train()
                case "Запуск в SUMO (модель)":
                    _loop_demo()
                case "SUMO без модели":
                    _loop_sumo_freerun()
                case "Сравнение режимов":
                    _loop_benchmark()
                case _:
                    pass
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
