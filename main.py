from __future__ import annotations

import argparse
import sys

from benchmark import run_comprehensive_benchmark
from demo import run_inference, run_sumo_freerun
from prepare_map import register_prepare_arguments, run_prepare_from_args
from src.simulation.env import REWARD_MODE_ALL, REWARD_MODES
from train import run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Единый CLI проекта управления трафиком")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Локальная карта → обрезка → SUMO .net.xml (без интернета)",
    )
    register_prepare_arguments(prepare_parser)

    train_parser = subparsers.add_parser("train", help="Обучение PPO")
    train_parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="SUMO-сеть (.net.xml), см. подготовку через prepare",
    )
    train_parser.add_argument(
        "--period", type=float, default=0.5, help="Интервал выпуска ТС (меньше — плотнее)"
    )
    train_parser.add_argument("--duration", type=int, default=3600, help="Длительность эпизода, с")
    train_parser.add_argument(
        "--with-pedestrians",
        action="store_true",
        help="Пешеходы в симуляции (нужна пешеходная сеть в .net.xml)",
    )
    train_parser.add_argument(
        "--pedestrian-period",
        type=float,
        default=None,
        metavar="SEC",
        help="Интервал пешеходов (по умолчанию от --period)",
    )
    train_parser.add_argument(
        "--with-public-transport",
        action="store_true",
        help="Автобусы (vclass=bus) как упрощённый ОТ",
    )
    train_parser.add_argument(
        "--public-transport-period",
        type=float,
        default=None,
        metavar="SEC",
        help="Интервал рейсов ОТ (по умолчанию от --period)",
    )
    train_parser.add_argument(
        "--reward-mode",
        type=str,
        choices=[*REWARD_MODES, REWARD_MODE_ALL],
        default="pressure",
        help="Функция награды агента TLS; all — подряд все режимы (свой ray_results на каждый)",
    )
    train_parser.add_argument(
        "--ped-reward-weight",
        type=float,
        default=None,
        metavar="W",
        help="Вес пешеходов для *_sidewalk (по умолчанию 1.0)",
    )
    train_parser.add_argument(
        "--sample-timeout-s",
        type=float,
        default=600.0,
        metavar="SEC",
        help="Таймаут сбора сэмплов с воркеров",
    )
    train_parser.add_argument(
        "--num-env-runners",
        type=int,
        default=6,
        metavar="N",
        help="Число параллельных env-воркеров Ray",
    )
    train_parser.add_argument(
        "--rollout-fragment-length",
        type=int,
        default=50,
        metavar="T",
        help="Длина фрагмента rollout",
    )
    train_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="DIR",
        help="Продолжить обучение: каталог trial или checkpoint_… в ray_results",
    )
    train_parser.add_argument(
        "--training-iterations",
        type=int,
        default=50,
        metavar="N",
        help="Остановка после N итераций обучения",
    )

    demo_parser = subparsers.add_parser("demo", help="Запуск обученной политики в SUMO")
    demo_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Каталог чекпоинта RLlib"
    )
    demo_parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="SUMO-сеть (.net.xml)",
    )
    demo_parser.add_argument(
        "--period", type=float, default=0.4, help="Интервал выпуска ТС (меньше — плотнее)"
    )
    demo_parser.add_argument("--duration", type=int, default=3600, help="Длительность, с")
    demo_parser.add_argument(
        "--with-pedestrians",
        action="store_true",
        help="Пешеходы в симуляции",
    )
    demo_parser.add_argument(
        "--pedestrian-period",
        type=float,
        default=None,
        metavar="SEC",
        help="Интервал пешеходов",
    )
    demo_parser.add_argument(
        "--with-public-transport",
        action="store_true",
        help="Автобусы (vclass=bus)",
    )
    demo_parser.add_argument(
        "--public-transport-period",
        type=float,
        default=None,
        metavar="SEC",
        help="Интервал ОТ",
    )
    demo_parser.add_argument(
        "--reward-mode",
        type=str,
        choices=list(REWARD_MODES),
        default="pressure",
        help="Функция награды среды (лог суммы награды в демо)",
    )
    demo_parser.add_argument(
        "--ped-reward-weight",
        type=float,
        default=None,
        metavar="W",
        help="Вес пешеходов для *_sidewalk",
    )
    demo_parser.add_argument(
        "--saliency",
        action="store_true",
        help="Включить saliency в консоль (эквивалент --explain-obs-every 1, если тот не задан)",
    )
    demo_parser.add_argument(
        "--explain-obs-every",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="STEPS",
        help="Демо: каждые STEPS шагов RL — топ saliency по наблюдению (градиент log π)",
    )
    demo_parser.add_argument(
        "--explain-tls-limit",
        type=int,
        default=2,
        metavar="N",
        help="Демо: не больше N светофоров для saliency за шаг",
    )
    demo_parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Демо: каталог для TensorBoard (скаляры saliency)",
    )
    demo_parser.add_argument(
        "--tensorboard-saliency-every",
        type=int,
        default=1,
        metavar="STEPS",
        help="Демо: при --tensorboard-dir писать saliency раз в STEPS",
    )
    demo_parser.add_argument("--no-gui", action="store_false", dest="gui", help="SUMO без GUI")
    demo_parser.set_defaults(gui=True)

    sumo_parser = subparsers.add_parser(
        "sumo",
        help="SUMO без RL: случайный трафик и программа светофоров из сети",
    )
    sumo_parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="Файл SUMO-сети (.net.xml)",
    )
    sumo_parser.add_argument("--period", type=float, default=0.4, help="Интервал выпуска ТС")
    sumo_parser.add_argument("--duration", type=int, default=3600, help="Длительность, с")
    sumo_parser.add_argument(
        "--with-pedestrians",
        action="store_true",
        help="Пешеходы в симуляции",
    )
    sumo_parser.add_argument("--pedestrian-period", type=float, default=None, metavar="SEC")
    sumo_parser.add_argument(
        "--with-public-transport",
        action="store_true",
        help="Автобусы (vclass=bus)",
    )
    sumo_parser.add_argument(
        "--public-transport-period",
        type=float,
        default=None,
        metavar="SEC",
    )
    sumo_parser.add_argument(
        "--reward-mode",
        type=str,
        choices=list(REWARD_MODES),
        default="pressure",
        help="Функция награды среды (для совместимости с MultiAgentTrafficEnv)",
    )
    sumo_parser.add_argument(
        "--ped-reward-weight",
        type=float,
        default=None,
        metavar="W",
        help="Вес пешеходов для *_sidewalk",
    )
    sumo_parser.add_argument("--no-gui", action="store_false", dest="gui", help="SUMO без GUI")
    sumo_parser.set_defaults(gui=True)

    bench_parser = subparsers.add_parser("benchmark", help="Сравнение с классическими TLS")
    bench_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Каталог чекпоинта RLlib"
    )
    bench_parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="Базовая SUMO-сеть (.net.xml)",
    )
    bench_parser.add_argument(
        "--period", type=float, default=0.4, help="Интервал выпуска ТС (меньше — плотнее)"
    )
    bench_parser.add_argument("--duration", type=int, default=3600, help="Длительность, с")
    bench_parser.add_argument(
        "--with-pedestrians",
        action="store_true",
        help="Пешеходы в симуляции",
    )
    bench_parser.add_argument(
        "--pedestrian-period",
        type=float,
        default=None,
        metavar="SEC",
        help="Интервал пешеходов",
    )
    bench_parser.add_argument(
        "--with-public-transport",
        action="store_true",
        help="Автобусы (vclass=bus)",
    )
    bench_parser.add_argument(
        "--public-transport-period",
        type=float,
        default=None,
        metavar="SEC",
        help="Интервал рейсов ОТ",
    )
    bench_parser.add_argument(
        "--reward-mode",
        type=str,
        choices=list(REWARD_MODES),
        default="pressure",
        help="Функция награды при оценке PPO",
    )
    bench_parser.add_argument(
        "--ped-reward-weight",
        type=float,
        default=None,
        metavar="W",
        help="Вес пешеходов для *_sidewalk",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Сохранить результаты benchmark в JSON",
    )
    bench_parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="TensorBoard: saliency для прогона PPO (фильтр benchmark/*)",
    )
    bench_parser.add_argument(
        "--tensorboard-saliency-every",
        type=int,
        default=1,
        metavar="STEPS",
        help="Benchmark: saliency в TensorBoard каждые N шагов RL (только PPO)",
    )
    bench_parser.add_argument(
        "--explain-tls-limit",
        type=int,
        default=2,
        metavar="N",
        help="Benchmark: не больше N TLS для saliency за шаг",
    )
    bench_parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Число независимых прогонов; при N>1 — среднее, СКО и доверительный интервал",
    )
    bench_parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        metavar="P",
        help="Уровень доверия для ДИ (по умолчанию 0.95)",
    )

    subparsers.add_parser("tui", help="Интерактивное меню")

    return parser


def main() -> None:
    from src.logging_config import configure_logging

    configure_logging()

    if len(sys.argv) == 1:
        from tui import run_tui

        run_tui()
        return

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        run_prepare_from_args(args)
    elif args.command == "train":
        run_train(
            args.map,
            args.period,
            args.duration,
            enable_pedestrians=args.with_pedestrians,
            pedestrian_period=args.pedestrian_period,
            enable_public_transport=args.with_public_transport,
            public_transport_period=args.public_transport_period,
            reward_mode=args.reward_mode,
            ped_reward_weight=args.ped_reward_weight,
            num_env_runners=args.num_env_runners,
            rollout_fragment_length=args.rollout_fragment_length,
            sample_timeout_s=args.sample_timeout_s,
            checkpoint_path=args.checkpoint,
            training_iterations=args.training_iterations,
        )
    elif args.command == "demo":
        explain_every = args.explain_obs_every
        tensorboard_dir = args.tensorboard_dir
        if getattr(args, "saliency", False):
            if explain_every is None:
                explain_every = 1
            if not tensorboard_dir:
                tensorboard_dir = "runs/saliency_demo"
        run_inference(
            args.checkpoint,
            args.map,
            args.duration,
            args.period,
            args.gui,
            enable_pedestrians=args.with_pedestrians,
            pedestrian_period=args.pedestrian_period,
            enable_public_transport=args.with_public_transport,
            public_transport_period=args.public_transport_period,
            reward_mode=args.reward_mode,
            ped_reward_weight=args.ped_reward_weight,
            explain_obs_every=explain_every,
            explain_tls_limit=args.explain_tls_limit,
            tensorboard_dir=tensorboard_dir,
            tensorboard_saliency_every=args.tensorboard_saliency_every,
        )
    elif args.command == "sumo":
        run_sumo_freerun(
            args.map,
            args.duration,
            args.period,
            args.gui,
            enable_pedestrians=args.with_pedestrians,
            pedestrian_period=args.pedestrian_period,
            enable_public_transport=args.with_public_transport,
            public_transport_period=args.public_transport_period,
            reward_mode=args.reward_mode,
            ped_reward_weight=args.ped_reward_weight,
        )
    elif args.command == "benchmark":
        import logging
        import os

        results = run_comprehensive_benchmark(
            args.checkpoint,
            args.map,
            args.duration,
            args.period,
            enable_pedestrians=args.with_pedestrians,
            pedestrian_period=args.pedestrian_period,
            enable_public_transport=args.with_public_transport,
            public_transport_period=args.public_transport_period,
            reward_mode=args.reward_mode,
            ped_reward_weight=args.ped_reward_weight,
            output_path=args.output,
            tensorboard_dir=args.tensorboard_dir,
            tensorboard_saliency_every=args.tensorboard_saliency_every,
            explain_tls_limit=args.explain_tls_limit,
            num_runs=args.runs,
            confidence_level=args.confidence_level,
        )
        if args.output and results is not None:
            logging.getLogger(__name__).info(
                "Результаты benchmark: %s", os.path.abspath(args.output)
            )
    elif args.command == "tui":
        from tui import run_tui

        run_tui()
    else:
        parser.error(f"Неизвестная команда: {args.command}")


if __name__ == "__main__":
    main()
