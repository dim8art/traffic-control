from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import subprocess
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
import traci
from scipy import stats
from ray.rllib.algorithms.algorithm import Algorithm

from src.rl.obs_salience import try_log_ppo_observation_saliency
from src.simulation.env import REWARD_MODES, MultiAgentTrafficEnv, traffic_env_config
from src.simulation.phase_control import (
    GreedyController,
    MaxPressureController,
    PhaseController,
)

logger = logging.getLogger(__name__)


class _PolicyLike(Protocol):
    def compute_single_action(
        self,
        obs: Any,
        *,
        policy_id: str = "traffic_policy",
        explore: bool = False,
    ) -> int: ...


def _resolve_checkpoint_dir(checkpoint_path: str) -> str:
    """Каталог trial (params.json) или конкретный checkpoint_000…"""
    path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if os.path.basename(path).startswith("checkpoint_"):
        return path
    if os.path.isfile(os.path.join(path, "rllib_checkpoint.json")):
        return path
    candidates = sorted(glob.glob(os.path.join(path, "checkpoint_*")))
    if candidates:
        return candidates[-1]
    return path


def _resolve_model_pt(checkpoint_path: str) -> str:
    ckpt_dir = _resolve_checkpoint_dir(checkpoint_path)
    if ckpt_dir.endswith("model.pt"):
        return ckpt_dir
    return os.path.join(ckpt_dir, "policies", "traffic_policy", "model", "model.pt")


class TorchPolicyInference:
    """Загрузка model.pt без algorithm_state.pkl (совместимость Python 3.10/3.11)."""

    def __init__(self, checkpoint_path: str) -> None:
        model_pt = _resolve_model_pt(checkpoint_path)
        if not os.path.isfile(model_pt):
            raise FileNotFoundError(f"Веса политики не найдены: {model_pt}")
        self._model = torch.load(model_pt, map_location="cpu", weights_only=False)
        self._model.eval()

    def compute_single_action(
        self,
        obs: Any,
        *,
        policy_id: str = "traffic_policy",
        explore: bool = False,
    ) -> int:
        del policy_id, explore
        arr = np.asarray(obs, dtype=np.float32)
        with torch.no_grad():
            logits, _ = self._model({"obs": torch.tensor(arr[None])})
        return int(logits.argmax(dim=-1).item())


def _resolve_explain_policy(algo: _PolicyLike) -> Any | None:
    """RLlib Policy для saliency; None, если загружен только model.pt без get_policy."""
    get_policy = getattr(algo, "get_policy", None)
    if not callable(get_policy):
        return None
    return get_policy("traffic_policy")


def _benchmark_tb_prefix(variant_label: str) -> str:
    slug = re.sub(r"[^\w]+", "_", variant_label.strip()).strip("_") or "variant"
    return f"benchmark/{slug}"


def load_policy(checkpoint_path: str) -> _PolicyLike:
    """Вывод через model.pt; полный Ray-чекпоинт — только если model.pt нет."""
    ckpt_dir = _resolve_checkpoint_dir(checkpoint_path)
    model_pt = _resolve_model_pt(checkpoint_path)
    if os.path.isfile(model_pt):
        logger.info(
            "Загрузка политики: %s (%s)",
            os.path.basename(ckpt_dir),
            model_pt,
        )
        return TorchPolicyInference(checkpoint_path)
    logger.info("model.pt не найден, пробуем полный чекпоинт Ray: %s", ckpt_dir)
    return Algorithm.from_checkpoint(ckpt_dir)


def generate_net_variant(base_net: str, variant_type: str, output_net: str) -> bool:
    """
    Использует netconvert для пересборки сети с другим типом TLS.
    Типы: 'static', 'actuated', 'delay_based'
    """
    logger.info("Генерация варианта сети TLS: %s", variant_type)
    try:
        # Утилита netconvert перенастраивает все светофоры в сети
        cmd = [
            "netconvert",
            "-s", base_net,
            "--tls.default-type", variant_type,
            "-o", output_net
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        logger.error("Не удалось сгенерировать %s: %s", variant_type, e)
        return False

def get_metrics(
    algo: _PolicyLike | None,
    sumo_net_xml_path: str,
    duration: int,
    period: float,
    *,
    is_ppo: bool = False,
    phase_controller: PhaseController | None = None,
    enable_pedestrians: bool = False,
    pedestrian_period: float | None = None,
    enable_public_transport: bool = False,
    public_transport_period: float | None = None,
    reward_mode: str = "pressure",
    ped_reward_weight: float | None = None,
    explain_policy: Any | None = None,
    tb_writer: Any | None = None,
    tb_prefix: str = "benchmark/ppo",
    tensorboard_saliency_every: int = 1,
    explain_tls_limit: int = 2,
) -> dict[str, float]:
    """Запуск симуляции для конкретного файла сети."""
    env_config = traffic_env_config(
        sumo_net_xml_path,
        traffic_period=period,
        duration=duration,
        gui=False,
        enable_pedestrians=enable_pedestrians,
        pedestrian_period=pedestrian_period,
        enable_public_transport=enable_public_transport,
        public_transport_period=public_transport_period,
        reward_mode=reward_mode,
        ped_reward_weight=ped_reward_weight,
        use_phase_actions=phase_controller is not None,
    )
    env = MultiAgentTrafficEnv(env_config)
    obs, info = env.reset()

    if phase_controller is not None:
        phase_controller.reset(list(obs.keys()))

    done = False
    total_reward = 0
    waiting_times = []
    speeds = []
    ep_tt_sum_car = 0.0
    ep_tt_cnt_car = 0
    ep_tt_sum_ped = 0.0
    ep_tt_cnt_ped = 0
    rl_step = 0
    tb_writes = 0
    tb_every = max(1, int(tensorboard_saliency_every))

    while not done:
        if phase_controller is not None:
            actions = phase_controller.compute_actions(list(obs.keys()))
        elif is_ppo and algo:
            actions = {
                aid: algo.compute_single_action(o, policy_id="traffic_policy", explore=False)
                for aid, o in obs.items()
            }
        else:
            # Для классических алгоритмов (Actuated и т.д.) просто позволяем SUMO работать
            actions = {aid: 0 for aid in obs.keys()}

        if (
            is_ppo
            and explain_policy is not None
            and tb_writer is not None
            and rl_step % tb_every == 0
        ):
            n_ok = try_log_ppo_observation_saliency(
                policy=explain_policy,
                obs=obs,
                actions=actions,
                global_step=rl_step,
                tb_writer=tb_writer,
                tb_prefix=tb_prefix,
                explain_tls_limit=explain_tls_limit,
                enable_pedestrians=enable_pedestrians,
                enable_public_transport=enable_public_transport,
            )
            if n_ok > 0:
                tb_writes += 1
            else:
                logger.warning(
                    "Benchmark saliency: на шаге %s не записано (градиент log π).",
                    rl_step,
                )

        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        if "__common__" in infos:
            common = infos["__common__"]
            waiting_times.append(common["system_mean_waiting_time"])
            speeds.append(common["system_avg_speed"])
            if rewards:
                total_reward += sum(rewards.values())
            ds = common.get("trip_completed_tt_sum_car")
            dc = common.get("trip_completed_count_car")
            if ds is not None and dc is not None:
                ep_tt_sum_car += float(ds)
                ep_tt_cnt_car += int(dc)
            ds = common.get("trip_completed_tt_sum_ped")
            dc = common.get("trip_completed_count_ped")
            if ds is not None and dc is not None:
                ep_tt_sum_ped += float(ds)
                ep_tt_cnt_ped += int(dc)

        done = terminated.get("__all__", False) or truncated.get("__all__", False)
        if traci.simulation.getTime() >= duration:
            done = True
        rl_step += 1

    traci.close()
    if tb_writer is not None and is_ppo:
        logger.info(
            "Benchmark saliency [%s]: записано %s шаг(ов), префикс TensorBoard «%s»",
            tb_prefix,
            tb_writes,
            tb_prefix,
        )
    metrics = {
        "Total Reward": total_reward,
        "Avg Speed (m/s)": np.mean(speeds) if speeds else 0,
        "Wait Time (s)": np.mean(waiting_times) if waiting_times else 0,
    }
    if ep_tt_cnt_car > 0:
        metrics["mean_completed_trip_time_car_s"] = ep_tt_sum_car / ep_tt_cnt_car
    if ep_tt_cnt_ped > 0:
        metrics["mean_completed_trip_time_pedestrian_s"] = ep_tt_sum_ped / ep_tt_cnt_ped
    return metrics


def _summarize_benchmark_runs(
    runs: list[dict[str, dict[str, float]]],
    *,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, dict[str, float]]]:
    """Среднее, СКО и двусторонний ДИ (t-распределение) по каждой метрике и режиму."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level должен быть в (0, 1)")
    variant_names = sorted({name for run in runs for name in run})
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for variant in variant_names:
        summary[variant] = {}
        metric_names = sorted(
            {mk for run in runs if variant in run for mk in run[variant]}
        )
        for metric in metric_names:
            values = [
                float(run[variant][metric])
                for run in runs
                if variant in run and metric in run[variant]
            ]
            n = len(values)
            if n == 0:
                continue
            arr = np.asarray(values, dtype=float)
            mean = float(arr.mean())
            if n == 1:
                summary[variant][metric] = {
                    "mean": mean,
                    "std": 0.0,
                    "ci_low": mean,
                    "ci_high": mean,
                    "n": 1.0,
                }
                continue
            std = float(arr.std(ddof=1))
            sem = float(stats.sem(arr))
            ci_low, ci_high = stats.t.interval(
                confidence_level,
                df=n - 1,
                loc=mean,
                scale=sem,
            )
            summary[variant][metric] = {
                "mean": mean,
                "std": std,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n": float(n),
            }
    return summary


def _summary_to_dataframe(
    summary: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for variant, metrics in summary.items():
        row: dict[str, float] = {}
        for metric, stats_row in metrics.items():
            row[f"{metric} mean"] = stats_row["mean"]
            row[f"{metric} std"] = stats_row["std"]
            row[f"{metric} ci_low"] = stats_row["ci_low"]
            row[f"{metric} ci_high"] = stats_row["ci_high"]
        rows[variant] = row
    return pd.DataFrame(rows).T.round(3)


def _log_benchmark_summary(
    runs: list[dict[str, dict[str, float]]],
    summary: dict[str, dict[str, dict[str, float]]],
    *,
    confidence_level: float,
) -> None:
    sep = "=" * 80
    pct = int(round(confidence_level * 100))
    logger.info(
        "%s\nИтог benchmark (%s прогон(ов), ДИ %s%%)\n%s",
        sep,
        len(runs),
        pct,
        sep,
    )
    logger.info("\n%s", _summary_to_dataframe(summary))
    for variant, metrics in summary.items():
        for metric, s in metrics.items():
            logger.info(
                "%s | %s: %.4f ± %.4f, ДИ [%s%%] [%.4f, %.4f] (n=%d)",
                variant,
                metric,
                s["mean"],
                s["std"],
                pct,
                s["ci_low"],
                s["ci_high"],
                int(s["n"]),
            )


def _execute_benchmark_variants(
    algo: _PolicyLike,
    base_sumo_net_xml: str,
    duration: int,
    period: float,
    *,
    enable_pedestrians: bool = False,
    pedestrian_period: float | None = None,
    enable_public_transport: bool = False,
    public_transport_period: float | None = None,
    reward_mode: str = "pressure",
    ped_reward_weight: float | None = None,
    explain_policy: Any | None = None,
    tb_writer: Any | None = None,
    tensorboard_saliency_every: int = 1,
    explain_tls_limit: int = 2,
) -> dict[str, dict[str, float]]:
    variants = {
        "static": {"name": "Static", "ppo": False, "phase_controller": None},
        "actuated": {"name": "Actuated", "ppo": False, "phase_controller": None},
        "delay_based": {"name": "Delay-based", "ppo": False, "phase_controller": None},
        "maxpressure": {
            "name": "MaxPressure",
            "ppo": False,
            "phase_controller": MaxPressureController(min_green_steps=1),
        },
        "greedy": {
            "name": "Greedy",
            "ppo": False,
            "phase_controller": GreedyController(min_green_steps=1),
        },
        "ppo": {"name": "PPO Agent (RL)", "ppo": True, "phase_controller": None},
    }

    results: dict[str, dict[str, float]] = {}
    generated_nets: list[str] = []

    for v_id, v_info in variants.items():
        logger.info("Тестирование: %s", v_info["name"])

        current_net = base_sumo_net_xml
        if v_id in ["actuated", "delay_based"]:
            variant_net = f"net_{v_id}.net.xml"
            if generate_net_variant(base_sumo_net_xml, v_id, variant_net):
                current_net = variant_net
                generated_nets.append(variant_net)
            else:
                logger.warning("Пропуск %s: пересборка сети не удалась.", v_id)
                continue

        res = get_metrics(
            algo,
            current_net,
            duration,
            period,
            is_ppo=v_info["ppo"],
            phase_controller=v_info.get("phase_controller"),
            enable_pedestrians=enable_pedestrians,
            pedestrian_period=pedestrian_period,
            enable_public_transport=enable_public_transport,
            public_transport_period=public_transport_period,
            reward_mode=reward_mode,
            ped_reward_weight=ped_reward_weight,
            explain_policy=explain_policy if v_info["ppo"] else None,
            tb_writer=tb_writer if v_info["ppo"] else None,
            tb_prefix=_benchmark_tb_prefix(v_info["name"]),
            tensorboard_saliency_every=tensorboard_saliency_every,
            explain_tls_limit=explain_tls_limit,
        )
        results[v_info["name"]] = res

    for net_path in generated_nets:
        if os.path.exists(net_path):
            os.remove(net_path)

    return results


def run_comprehensive_benchmark(
    checkpoint_path: str,
    base_sumo_net_xml: str,
    duration: int,
    period: float,
    *,
    enable_pedestrians: bool = False,
    pedestrian_period: float | None = None,
    enable_public_transport: bool = False,
    public_transport_period: float | None = None,
    reward_mode: str = "pressure",
    ped_reward_weight: float | None = None,
    output_path: str | None = None,
    tensorboard_dir: str | None = None,
    tensorboard_saliency_every: int = 1,
    explain_tls_limit: int = 2,
    num_runs: int = 1,
    confidence_level: float = 0.95,
) -> dict[str, Any] | None:
    if num_runs < 1:
        raise ValueError("num_runs должен быть >= 1")

    algo = load_policy(checkpoint_path)

    tb_writer = None
    tb_log_dir: str | None = None
    explain_policy = None
    if tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "Для --tensorboard-dir установите зависимость: pip install tensorboard"
            ) from exc
        tb_log_dir = os.path.abspath(tensorboard_dir)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        tb_writer.add_scalar("benchmark/_meta/writer_ready", 1.0, 0)
        tb_writer.flush()
        explain_policy = _resolve_explain_policy(algo)
        if explain_policy is None:
            logger.warning(
                "Saliency benchmark: нужен полный Ray-чекпоинт (Algorithm), не только model.pt."
            )
        else:
            logger.info(
                "TensorBoard benchmark saliency → %s (только прогон PPO; фильтр «benchmark»).\n"
                "  tensorboard --logdir %s",
                tb_log_dir,
                tb_log_dir,
            )

    all_runs: list[dict[str, dict[str, float]]] = []
    for run_idx in range(num_runs):
        if num_runs > 1:
            logger.info("=== Benchmark прогон %s/%s ===", run_idx + 1, num_runs)
        use_tb = tb_writer if (num_runs == 1 or run_idx == 0) else None
        run_results = _execute_benchmark_variants(
            algo,
            base_sumo_net_xml,
            duration,
            period,
            enable_pedestrians=enable_pedestrians,
            pedestrian_period=pedestrian_period,
            enable_public_transport=enable_public_transport,
            public_transport_period=public_transport_period,
            reward_mode=reward_mode,
            ped_reward_weight=ped_reward_weight,
            explain_policy=explain_policy if use_tb is not None else None,
            tb_writer=use_tb,
            tensorboard_saliency_every=tensorboard_saliency_every,
            explain_tls_limit=explain_tls_limit,
        )
        all_runs.append(run_results)
        if output_path and num_runs == 1:
            write_benchmark_results(run_results, output_path)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    if num_runs == 1:
        results = all_runs[0]
        df = pd.DataFrame(results).T.round(3)
        sep = "=" * 80
        logger.info("%s\nИтог benchmark (пересборка сетей)\n%s\n\n%s", sep, sep, df)
        return results

    summary = _summarize_benchmark_runs(all_runs, confidence_level=confidence_level)
    _log_benchmark_summary(all_runs, summary, confidence_level=confidence_level)
    payload = {
        "num_runs": num_runs,
        "confidence_level": confidence_level,
        "runs": all_runs,
        "summary": summary,
    }
    if output_path:
        write_benchmark_results(payload, output_path)
    return payload


def _to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def write_benchmark_results(results: dict[str, Any], output_path: str) -> None:
    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(results), f, indent=2, ensure_ascii=False)


def _write_results_json(results: dict, output_path: str) -> None:
    write_benchmark_results(results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнение PPO с классическими режимами TLS")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Каталог чекпоинта RLlib"
    )
    parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="Базовая SUMO-сеть (.net.xml) для сравнения",
    )
    parser.add_argument("--duration", type=int, default=3600, help="Длительность, с")
    parser.add_argument(
        "--period", type=float, default=0.4, help="Интервал выпуска ТС (меньше — плотнее)"
    )
    parser.add_argument(
        "--with-pedestrians",
        action="store_true",
        help="Пешеходы в симуляции (см. train.py)",
    )
    parser.add_argument(
        "--pedestrian-period", type=float, default=None, metavar="SEC", help="Интервал пешеходов"
    )
    parser.add_argument(
        "--with-public-transport",
        action="store_true",
        help="Автобусы (vclass=bus)",
    )
    parser.add_argument(
        "--public-transport-period",
        type=float,
        default=None,
        metavar="SEC",
        help="Интервал рейсов ОТ",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=list(REWARD_MODES),
        default="pressure",
        help="Функция награды при оценке PPO",
    )
    parser.add_argument(
        "--ped-reward-weight",
        type=float,
        default=None,
        metavar="W",
        help="Вес пешеходов для *_sidewalk",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Сохранить результаты в JSON",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="TensorBoard: saliency только для прогона PPO (префикс benchmark/<режим>)",
    )
    parser.add_argument(
        "--tensorboard-saliency-every",
        type=int,
        default=1,
        metavar="STEPS",
        help="Запись saliency в TensorBoard каждые N шагов RL (только PPO)",
    )
    parser.add_argument(
        "--explain-tls-limit",
        type=int,
        default=2,
        metavar="N",
        help="Не больше N светофоров для saliency за шаг (только PPO)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Число независимых прогонов (случайный трафик); при N>1 — среднее, СКО и ДИ",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        metavar="P",
        help="Уровень доверия для интервала (по умолчанию 0.95)",
    )
    args = parser.parse_args()

    from src.logging_config import configure_logging

    configure_logging()

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
        logger.info("Результаты записаны в %s", os.path.abspath(args.output))