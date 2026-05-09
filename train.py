from __future__ import annotations

import argparse
import os

import numpy as np
from gymnasium import spaces

from ray import tune
from ray.tune import CheckpointConfig, RunConfig
from ray.rllib.algorithms.ppo import PPOConfig

from src.simulation.env import MultiAgentTrafficEnv, TrafficCallbacks


def run_train(sumo_net_xml_path: str, traffic_period: float, duration: int) -> None:
    if not os.path.exists(sumo_net_xml_path):
        raise FileNotFoundError(f"SUMO network not found: {sumo_net_xml_path}")

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .environment(
            MultiAgentTrafficEnv,
            env_config={
                "net_file": sumo_net_xml_path,
                "traffic_period": traffic_period,  # Pass period (e.g., 0.2 for heavy traffic)
                "duration": duration,  # Simulation time (e.g., 3600s)
            },
        )
        .framework("torch")
        .env_runners(
            num_env_runners=6, rollout_fragment_length=50, sample_timeout_s=120
        )
        .callbacks(TrafficCallbacks)
        .multi_agent(
            policies={
                "traffic_policy": (
                    None,
                    spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32),
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
        help="Traffic generation period (lower = more traffic, e.g. 0.2)"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=3600, 
        help="Simulation duration in seconds"
    )
    
    args = parser.parse_args()

    from src.logging_config import configure_logging

    configure_logging()

    run_train(args.map, args.period, args.duration)