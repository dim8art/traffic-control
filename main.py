from __future__ import annotations

import argparse
import sys

from benchmark import run_comprehensive_benchmark
from demo import run_inference, run_sumo_freerun
from prepare_map import register_prepare_arguments, run_prepare_from_args
from train import run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic Control project unified CLI")
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
    train_parser.add_argument("--period", type=float, default=0.5, help="Traffic period")
    train_parser.add_argument("--duration", type=int, default=3600, help="Episode duration")
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

    demo_parser = subparsers.add_parser("demo", help="Запуск обученной политики в SUMO")
    demo_parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    demo_parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="SUMO-сеть (.net.xml)",
    )
    demo_parser.add_argument("--period", type=float, default=0.4, help="Traffic period")
    demo_parser.add_argument("--duration", type=int, default=3600, help="Episode duration")
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
        help="Демо: каждые STEPS шагов RL — топ saliency по наблюдению (grad log π)",
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
        help="Демо: каталог для TensorBoard (saliency Scalars)",
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
        help=".net.xml",
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
    sumo_parser.add_argument("--no-gui", action="store_false", dest="gui", help="SUMO без GUI")
    sumo_parser.set_defaults(gui=True)

    bench_parser = subparsers.add_parser("benchmark", help="Сравнение с классическими TLS")
    bench_parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    bench_parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="Базовая SUMO-сеть (.net.xml)",
    )
    bench_parser.add_argument("--period", type=float, default=0.4, help="Traffic period")
    bench_parser.add_argument("--duration", type=int, default=3600, help="Episode duration")
    bench_parser.add_argument("--with-pedestrians", action="store_true")
    bench_parser.add_argument("--pedestrian-period", type=float, default=None, metavar="SEC")
    bench_parser.add_argument("--with-public-transport", action="store_true")
    bench_parser.add_argument(
        "--public-transport-period",
        type=float,
        default=None,
        metavar="SEC",
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
        )
    elif args.command == "demo":
        explain_every = args.explain_obs_every
        if getattr(args, "saliency", False) and explain_every is None:
            explain_every = 1
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
            explain_obs_every=explain_every,
            explain_tls_limit=args.explain_tls_limit,
            tensorboard_dir=args.tensorboard_dir,
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
        )
    elif args.command == "benchmark":
        run_comprehensive_benchmark(
            args.checkpoint,
            args.map,
            args.duration,
            args.period,
            enable_pedestrians=args.with_pedestrians,
            pedestrian_period=args.pedestrian_period,
            enable_public_transport=args.with_public_transport,
            public_transport_period=args.public_transport_period,
        )
    elif args.command == "tui":
        from tui import run_tui

        run_tui()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
