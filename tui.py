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


def _prompt_simulation_traffic_extras(
    *, traffic_period: float = 0.5
) -> dict[str, object] | None:
    """
    Пешеходы / ОТ — те же опции, что --with-pedestrians и --with-public-transport в CLI.
    Возвращает None, если пользователь прервал ввод.
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
        ped_period = _prompt_float(
            "Интервал пешеходов, с",
            f"{max(traffic_period * 2.0, 1.0):g}",
        )

    pt_period: float | None = None
    if pt:
        pt_period = _prompt_float(
            "Интервал рейсов ОТ, с",
            f"{max(traffic_period * 3.0, 2.0):g}",
        )

    return {
        "enable_pedestrians": bool(ped),
        "enable_public_transport": bool(pt),
        "pedestrian_period": ped_period,
        "public_transport_period": pt_period,
    }


def _prompt_reward_extras() -> dict[str, object] | None:
    """Режим награды reward_mode и опционально ped_reward_weight для *_sidewalk."""
    from src.simulation.env import REWARD_MODE_ALL, REWARD_MODES

    choices = [
        Choice(title="Все режимы подряд", value=REWARD_MODE_ALL),
        *[Choice(title=rm, value=rm) for rm in REWARD_MODES],
    ]
    rm = questionary.select(
        "Режим награды",
        choices=choices,
        default="pressure",
    ).ask()
    if rm is None:
        return None
    ped_w: float | None = None
    if rm == REWARD_MODE_ALL:
        raw = _prompt_text(
            "Вес пешеходов для *_sidewalk (пусто — 1.0 по умолчанию)", ""
        )
        if raw and raw.strip():
            try:
                ped_w = float(raw.strip())
            except ValueError:
                questionary.print("Не число, для sidewalk-режимов будет 1.0.")
    elif "sidewalk" in str(rm):
        ped_w = _prompt_float("Вес пешеходов в награде", "1.0")
    return {"reward_mode": rm, "ped_reward_weight": ped_w}


def _prompt_demo_saliency() -> dict[str, object] | None:
    """Опции saliency для run_inference (консоль и TensorBoard независимо)."""
    out: dict[str, object] = {
        "explain_obs_every": None,
        "explain_tls_limit": 2,
        "tensorboard_dir": None,
        "tensorboard_saliency_every": 1,
    }
    tb_dir = (
        _prompt_text(
            "Каталог TensorBoard для saliency (пусто — не писать)",
            "runs/saliency_demo",
        )
        or ""
    ).strip()
    if tb_dir:
        out["tensorboard_dir"] = tb_dir
        out["tensorboard_saliency_every"] = _prompt_int(
            "TensorBoard: каждые N шагов RL", "1"
        )

    console_every = _prompt_int("Консоль saliency: каждые N шагов RL (0 — выкл.)", "0")
    if console_every > 0:
        out["explain_obs_every"] = console_every
        out["explain_tls_limit"] = _prompt_int(
            "Не больше скольких светофоров выводить за шаг", "2"
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
    training_iterations = _prompt_int("Число итераций обучения", "50")
    traffic_extras = _prompt_simulation_traffic_extras(traffic_period=period)
    if traffic_extras is None:
        return
    reward_extras = _prompt_reward_extras()
    if reward_extras is None:
        return

    from src.simulation.env import REWARD_MODE_ALL

    checkpoint_path: str | None = None
    resume = questionary.confirm(
        "Продолжить обучение с чекпоинта (Ray Tune)?",
        default=False,
    ).ask()
    if resume is None:
        return
    if resume:
        if reward_extras.get("reward_mode") == REWARD_MODE_ALL:
            questionary.print("Режим «все»: продолжение с чекпоинта недоступно.")
        else:
            ckpt = _prompt_existing_path(
                "Каталог trial (PPO_…) или checkpoint_000… в ray_results"
            )
            if not ckpt:
                questionary.print("Путь не найден.")
                return
            checkpoint_path = ckpt

    if reward_extras.get("reward_mode") == REWARD_MODE_ALL:
        from src.simulation.env import REWARD_MODES

        questionary.print(
            f"Обучение по всем режимам ({len(REWARD_MODES)} прогонов), у каждого свой каталог…"
        )
    else:
        questionary.print("Обучение…")
    run_train(
        net_xml,
        period,
        duration,
        checkpoint_path=checkpoint_path,
        training_iterations=training_iterations,
        **traffic_extras,
        **reward_extras,
    )


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
    traffic_extras = _prompt_simulation_traffic_extras(traffic_period=period)
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

    ckpt = _prompt_existing_path("Каталог чекпоинта")
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
    traffic_extras = _prompt_simulation_traffic_extras(traffic_period=period)
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

    ckpt = _prompt_existing_path("Каталог чекпоинта")
    if not ckpt or not os.path.isdir(ckpt):
        questionary.print("Укажите существующий каталог.")
        return
    net_xml = _prompt_existing_path("Сеть SUMO (.net.xml)")
    if not net_xml:
        return
    period = _prompt_float("Интервал выпуска ТС", "0.4")
    duration = _prompt_int("Длительность, с", "3600")
    traffic_extras = _prompt_simulation_traffic_extras(traffic_period=period)
    if traffic_extras is None:
        return
    reward_extras = _prompt_reward_extras()
    if reward_extras is None:
        return
    saliency_opts = _prompt_demo_saliency()
    if saliency_opts is None:
        return
    num_runs = _prompt_int("Число прогонов benchmark", "1")
    if num_runs < 1:
        questionary.print("Число прогонов должно быть >= 1.")
        return
    confidence_level = 0.95
    if num_runs > 1:
        confidence_level = _prompt_float("Уровень доверия для ДИ (0–1)", "0.95")
        if not 0 < confidence_level < 1:
            questionary.print("Уровень доверия должен быть в (0, 1), используется 0.95.")
            confidence_level = 0.95
    output_path: str | None = None
    json_default = (
        f"benchmark_{num_runs}runs.json" if num_runs > 1 else "benchmark_results.json"
    )
    save_json = questionary.confirm(
        "Сохранить результаты в JSON?",
        default=num_runs > 1,
    ).ask()
    if save_json is None:
        return
    if save_json:
        raw = _prompt_text("Путь к JSON", json_default)
        output_path = _abspath(raw) if raw and raw.strip() else _abspath(json_default)
    questionary.print("Сравнение…")
    result = run_comprehensive_benchmark(
        ckpt,
        net_xml,
        duration,
        period,
        output_path=output_path,
        tensorboard_dir=saliency_opts.get("tensorboard_dir"),
        tensorboard_saliency_every=int(saliency_opts.get("tensorboard_saliency_every", 1)),
        explain_tls_limit=int(saliency_opts.get("explain_tls_limit", 2)),
        num_runs=num_runs,
        confidence_level=confidence_level,
        **traffic_extras,
        **reward_extras,
    )
    if output_path:
        questionary.print(f"Результаты сохранены: {output_path}")
    if num_runs > 1 and isinstance(result, dict) and "summary" in result:
        from benchmark import _summary_to_dataframe

        questionary.print(
            f"\nСводка ({num_runs} прогонов, ДИ {int(round(confidence_level * 100))}%):"
        )
        questionary.print(_summary_to_dataframe(result["summary"]).to_string())


def run_tui() -> None:
    while True:
        choice = questionary.select(
            "Управление трафиком",
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
