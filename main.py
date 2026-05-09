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

    subparsers.add_parser("tui", help="Интерактивное меню")

    return parser


def main() -> None:
    if len(sys.argv) == 1:
        from tui import run_tui

        run_tui()
        return

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        run_prepare_from_args(args)
    elif args.command == "train":
        run_train(args.map, args.period, args.duration)
    elif args.command == "demo":
        run_inference(args.checkpoint, args.map, args.duration, args.period, args.gui)
    elif args.command == "sumo":
        run_sumo_freerun(args.map, args.duration, args.period, args.gui)
    elif args.command == "benchmark":
        run_comprehensive_benchmark(
            args.checkpoint, args.map, args.duration, args.period
        )
    elif args.command == "tui":
        from tui import run_tui

        run_tui()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
