from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
import traci
from ray.rllib.algorithms.algorithm import Algorithm

from src.simulation.env import REWARD_MODES, MultiAgentTrafficEnv, traffic_env_config

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
            raise FileNotFoundError(f"Policy weights not found: {model_pt}")
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


def load_policy(checkpoint_path: str) -> _PolicyLike:
    """Inference через model.pt; полный Ray-чекпоинт — только если model.pt нет."""
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
    enable_pedestrians: bool = False,
    pedestrian_period: float | None = None,
    enable_public_transport: bool = False,
    public_transport_period: float | None = None,
    reward_mode: str = "pressure",
    ped_reward_weight: float | None = None,
) -> dict[str, float]:
    """Запуск симуляции для конкретного файла сети"""
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
    )
    env = MultiAgentTrafficEnv(env_config)
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    waiting_times = []
    speeds = []
    ep_tt_sum_car = 0.0
    ep_tt_cnt_car = 0
    ep_tt_sum_ped = 0.0
    ep_tt_cnt_ped = 0

    while not done:
        if is_ppo and algo:
            actions = {
                aid: algo.compute_single_action(o, policy_id="traffic_policy", explore=False)
                for aid, o in obs.items()
            }
        else:
            # Для классических алгоритмов (Actuated и т.д.) просто позволяем SUMO работать
            actions = {aid: 0 for aid in obs.keys()} 

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

    traci.close()
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
) -> dict[str, dict[str, float]] | None:
    algo = load_policy(checkpoint_path)
    
    # Определяем варианты для тестирования
    # SOTL в netconvert напрямую не задается (требует сложной настройки детекторов),
    # поэтому ограничимся 3 классическими типами + PPO.
    variants = {
        "static": {"name": "Static (Fixed)", "ppo": False},
        "actuated": {"name": "Actuated (Sensors)", "ppo": False},
        "delay_based": {"name": "Delay-based", "ppo": False},
        "ppo": {"name": "PPO Agent (RL)", "ppo": True}
    }

    results = {}

    for v_id, v_info in variants.items():
        logger.info("Тестирование: %s", v_info["name"])

        # Для PPO и Static используем базовую сеть, для остальных — генерируем новую
        current_net = base_sumo_net_xml
        if v_id in ["actuated", "delay_based"]:
            variant_net = f"net_{v_id}.net.xml"
            if generate_net_variant(base_sumo_net_xml, v_id, variant_net):
                current_net = variant_net
            else:
                logger.warning("Пропуск %s: пересборка сети не удалась.", v_id)
                continue

        res = get_metrics(
            algo,
            current_net,
            duration,
            period,
            is_ppo=v_info["ppo"],
            enable_pedestrians=enable_pedestrians,
            pedestrian_period=pedestrian_period,
            enable_public_transport=enable_public_transport,
            public_transport_period=public_transport_period,
            reward_mode=reward_mode,
            ped_reward_weight=ped_reward_weight,
        )
        results[v_info["name"]] = res
        if output_path:
            _write_results_json(results, output_path)

    for v_id in ["actuated", "delay_based"]:
        f = f"net_{v_id}.net.xml"
        if os.path.exists(f):
            os.remove(f)

    df = pd.DataFrame(results).T.round(3)
    sep = "=" * 80
    logger.info("%s\nИтог benchmark (пересборка сетей)\n%s\n\n%s", sep, sep, df)
    return results


def _write_results_json(results: dict, output_path: str) -> None:
    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    serializable = {
        k: {
            mk: float(mv) if isinstance(mv, (np.floating, np.integer)) else mv
            for mk, mv in v.items()
        }
        for k, v in results.items()
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="Базовая SUMO-сеть (.net.xml) для сравнения",
    )
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--period", type=float, default=0.4)
    parser.add_argument(
        "--with-pedestrians",
        action="store_true",
        help="Пешеходы в симуляции (см. train.py)",
    )
    parser.add_argument("--pedestrian-period", type=float, default=None, metavar="SEC")
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
    )
    if args.output and results is not None:
        logger.info("Результаты записаны в %s", os.path.abspath(args.output))