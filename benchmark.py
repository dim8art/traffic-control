from __future__ import annotations

import argparse
import logging
import os
import subprocess

import numpy as np
import pandas as pd
import ray
import traci
from ray.rllib.algorithms.algorithm import Algorithm

from src.simulation.env import MultiAgentTrafficEnv, traffic_env_config

logger = logging.getLogger(__name__)


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
    algo: Algorithm | None,
    sumo_net_xml_path: str,
    duration: int,
    period: float,
    *,
    is_ppo: bool = False,
    enable_pedestrians: bool = False,
    pedestrian_period: float | None = None,
    enable_public_transport: bool = False,
    public_transport_period: float | None = None,
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
    )
    env = MultiAgentTrafficEnv(env_config)
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    waiting_times = []
    speeds = []

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
            waiting_times.append(infos["__common__"]["system_mean_waiting_time"])
            speeds.append(infos["__common__"]["system_avg_speed"])
            if rewards:
                total_reward += sum(rewards.values())

        done = terminated.get("__all__", False) or truncated.get("__all__", False)
    
    traci.close()
    return {
        "Total Reward": total_reward,
        "Avg Speed (m/s)": np.mean(speeds) if speeds else 0,
        "Wait Time (s)": np.mean(waiting_times) if waiting_times else 0
    }

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
) -> None:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR")

    algo = Algorithm.from_checkpoint(checkpoint_path)
    
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
        )
        results[v_info['name']] = res

    # Вывод итогов
    df = pd.DataFrame(results).T.round(3)
    sep = "=" * 80
    logger.info("%s\nИтог benchmark (пересборка сетей)\n%s\n\n%s", sep, sep, df)
    
    # Очистка временных файлов
    for v_id in ["actuated", "delay_based"]:
        f = f"net_{v_id}.net.xml"
        if os.path.exists(f): os.remove(f)

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
    args = parser.parse_args()

    from src.logging_config import configure_logging

    configure_logging()

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