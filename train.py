from __future__ import annotations

import argparse
import os

from gymnasium import spaces

from ray import tune
from ray.tune import CheckpointConfig, RunConfig
from ray.rllib.algorithms.ppo import PPOConfig

from src.simulation.env import (
    REWARD_MODES,
    MultiAgentTrafficEnv,
    TrafficCallbacks,
    traffic_env_config,
    traffic_policy_observation_space,
)


def run_train(
    sumo_net_xml_path: str,
    traffic_period: float,
    duration: int,
    *,
    enable_pedestrians: bool = False,
    pedestrian_period: float | None = None,
    enable_public_transport: bool = False,
    public_transport_period: float | None = None,
    reward_mode: str = "pressure",
    ped_reward_weight: float | None = None,
    num_env_runners: int = 6,
    rollout_fragment_length: int = 50,
    sample_timeout_s: float = 600.0,
) -> None:
    if not os.path.exists(sumo_net_xml_path):
        raise FileNotFoundError(f"SUMO network not found: {sumo_net_xml_path}")

    env_cfg = traffic_env_config(
        sumo_net_xml_path,
        traffic_period=traffic_period,
        duration=duration,
        enable_pedestrians=enable_pedestrians,
        pedestrian_period=pedestrian_period,
        enable_public_transport=enable_public_transport,
        public_transport_period=public_transport_period,
        reward_mode=reward_mode,
        ped_reward_weight=ped_reward_weight,
    )
    obs_space = traffic_policy_observation_space(
        enable_pedestrians=enable_pedestrians,
        enable_public_transport=enable_public_transport,
    )

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .environment(
            MultiAgentTrafficEnv,
            env_config=env_cfg,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=rollout_fragment_length,
            sample_timeout_s=sample_timeout_s,
        )
        .callbacks(TrafficCallbacks)
        .multi_agent(
            policies={
                "traffic_policy": (
                    None,
                    obs_space,
                    spaces.Discrete(2),
                    {},
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "traffic_policy",
        ).checkpointing(
            export_native_model_files=True,
            checkpoint_trainable_policies_only=False,
        )
    )

    config.train_batch_size = 4000
    config.sgd_minibatch_size = 256
    config.num_sgd_iter = 10
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            name="SUMO_PPO_FINAL",
            stop={"training_iteration": 1000},
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
        ),
    )
    tuner.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start RL training for SUMO agent")
    parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="Путь к SUMO-сети (.net.xml), полученной через prepare",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=0.5,
        help="Traffic generation period (lower = more traffic, e.g. 0.2)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Simulation duration in seconds",
    )
    parser.add_argument(
        "--with-pedestrians",
        action="store_true",
        help="Добавить пешеходов (randomTrips --pedestrians; нужны тротуары в сети)",
    )
    parser.add_argument(
        "--pedestrian-period",
        type=float,
        default=None,
        metavar="SEC",
        help="Интервал между появлением пешеходов (по умолчанию ~2× --period)",
    )
    parser.add_argument(
        "--with-public-transport",
        action="store_true",
        help="Добавить автобусы vclass=bus как упрощённый ОТ",
    )
    parser.add_argument(
        "--public-transport-period",
        type=float,
        default=None,
        metavar="SEC",
        help="Интервал рейсов ОТ (по умолчанию ~3× --period)",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=list(REWARD_MODES),
        default="pressure",
        help="Функция награды (в т.ч. *_sidewalk — см. src/simulation/env.py)",
    )
    parser.add_argument(
        "--ped-reward-weight",
        type=float,
        default=None,
        metavar="W",
        help="Вес пешеходов для *_sidewalk (по умолчанию 1.0)",
    )
    parser.add_argument(
        "--sample-timeout-s",
        type=float,
        default=600.0,
        metavar="SEC",
        help="Таймаут сбора сэмплов с env-воркеров Ray",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=6,
        metavar="N",
        help="Число параллельных env-воркеров",
    )
    parser.add_argument(
        "--rollout-fragment-length",
        type=int,
        default=50,
        metavar="T",
        help="Длина фрагмента rollout на воркер",
    )

    args = parser.parse_args()

    from src.logging_config import configure_logging

    configure_logging()

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
    )
