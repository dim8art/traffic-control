from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime

from gymnasium import spaces

from ray import tune
from ray.tune import CheckpointConfig, RunConfig, TuneConfig
from ray.rllib.algorithms.ppo import PPOConfig

from src.simulation.env import (
    REWARD_MODE_ALL,
    REWARD_MODES,
    MultiAgentTrafficEnv,
    TrafficCallbacks,
    traffic_env_config,
    traffic_policy_observation_space,
)

logger = logging.getLogger(__name__)

RAY_RESULTS_ROOT = "ray_results"


def _slug_part(text: str, *, max_len: int = 48) -> str:
    out = re.sub(r"[^\w.-]+", "-", str(text).strip())
    out = re.sub(r"-+", "-", out).strip("-")
    return out[:max_len] or "x"


def _fmt_num(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _fmt_optional_period(value: float | None) -> str:
    return "auto" if value is None else _fmt_num(value)


def build_training_run_name(
    sumo_net_xml_path: str,
    traffic_period: float,
    duration: int,
    *,
    training_iterations: int,
    reward_mode: str,
    enable_pedestrians: bool,
    enable_public_transport: bool,
    pedestrian_period: float | None,
    public_transport_period: float | None,
    ped_reward_weight: float | None,
    num_env_runners: int,
    rollout_fragment_length: int,
    started_at: datetime | None = None,
) -> str:
    """Имя эксперимента Ray Tune: дата-время и ключевые параметры запуска."""
    ts = (started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    map_stem = os.path.splitext(os.path.basename(sumo_net_xml_path))[0]
    if map_stem.endswith(".net"):
        map_stem = map_stem[:-4]
    map_name = _slug_part(map_stem, max_len=32)
    parts = [
        ts,
        map_name,
        f"p{_fmt_num(traffic_period)}",
        f"d{duration}",
        f"rm{_slug_part(reward_mode, max_len=24)}",
        f"iter{training_iterations}",
        f"env{num_env_runners}",
        f"rf{rollout_fragment_length}",
    ]
    if enable_pedestrians:
        parts.append(f"ped{_fmt_optional_period(pedestrian_period)}")
    if enable_public_transport:
        parts.append(f"bus{_fmt_optional_period(public_transport_period)}")
    if ped_reward_weight is not None:
        parts.append(f"pw{_fmt_num(ped_reward_weight)}")
    return "_".join(parts)


def _resolve_tune_restore_dir(path: str) -> str:
    """
    Путь для tune.Tuner.restore: каталог trial (с params.json) или checkpoint_000…
    внутри trial.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Путь к чекпоинту не найден: {path}")

    if os.path.basename(path).startswith("checkpoint_"):
        path = os.path.dirname(path)

    cur = path
    while True:
        if os.path.isfile(os.path.join(cur, "params.json")) or os.path.isfile(
            os.path.join(cur, "params.pkl")
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    return path


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
    checkpoint_path: str | None = None,
    training_iterations: int = 50,
) -> None:
    if reward_mode == REWARD_MODE_ALL:
        if checkpoint_path:
            logger.warning(
                "Режим «all»: --checkpoint игнорируется — для каждого reward_mode свой каталог в ray_results.",
            )
        total = len(REWARD_MODES)
        for index, mode in enumerate(REWARD_MODES, start=1):
            logger.info(
                "Обучение %s/%s, reward_mode=%s",
                index,
                total,
                mode,
            )
            run_train(
                sumo_net_xml_path,
                traffic_period,
                duration,
                enable_pedestrians=enable_pedestrians,
                pedestrian_period=pedestrian_period,
                enable_public_transport=enable_public_transport,
                public_transport_period=public_transport_period,
                reward_mode=mode,
                ped_reward_weight=(
                    ped_reward_weight if "sidewalk" in mode else None
                ),
                num_env_runners=num_env_runners,
                rollout_fragment_length=rollout_fragment_length,
                sample_timeout_s=sample_timeout_s,
                checkpoint_path=None,
                training_iterations=training_iterations,
            )
        logger.info("Обучение по всем режимам награды завершено (%s прогонов).", total)
        return

    if not os.path.exists(sumo_net_xml_path):
        raise FileNotFoundError(f"SUMO-сеть не найдена: {sumo_net_xml_path}")

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

    checkpoint_cfg = CheckpointConfig(
        num_to_keep=3,
        checkpoint_frequency=1,
        checkpoint_at_end=True,
    )
    tune_config = TuneConfig(
        num_samples=1,
        trial_dirname_creator=lambda _: "train",
    )
    param_space = config.to_dict()

    if checkpoint_path:
        restore_dir = _resolve_tune_restore_dir(checkpoint_path)
        logger.info("Продолжение обучения из %s", restore_dir)
        run_config = RunConfig(
            stop={"training_iteration": training_iterations},
            checkpoint_config=checkpoint_cfg,
        )
        tuner = tune.Tuner.restore(
            restore_dir,
            trainable="PPO",
            param_space=param_space,
            run_config=run_config,
            tune_config=tune_config,
        )
    else:
        run_name = build_training_run_name(
            sumo_net_xml_path,
            traffic_period,
            duration,
            training_iterations=training_iterations,
            reward_mode=reward_mode,
            enable_pedestrians=enable_pedestrians,
            enable_public_transport=enable_public_transport,
            pedestrian_period=pedestrian_period,
            public_transport_period=public_transport_period,
            ped_reward_weight=ped_reward_weight,
            num_env_runners=num_env_runners,
            rollout_fragment_length=rollout_fragment_length,
        )
        storage_root = os.path.abspath(RAY_RESULTS_ROOT)
        os.makedirs(storage_root, exist_ok=True)
        logger.info(
            "Каталог обучения и чекпоинтов: %s/%s/train/checkpoint_*",
            storage_root,
            run_name,
        )
        run_config = RunConfig(
            name=run_name,
            storage_path=storage_root,
            stop={"training_iteration": training_iterations},
            checkpoint_config=checkpoint_cfg,
        )
        tuner = tune.Tuner(
            "PPO",
            param_space=param_space,
            run_config=run_config,
            tune_config=tune_config,
        )
    tuner.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение RL-агента для SUMO")
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
        help="Интервал генерации трафика (меньше — плотнее, напр. 0.2)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Длительность симуляции, с",
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
        choices=[*REWARD_MODES, REWARD_MODE_ALL],
        default="pressure",
        help="Функция награды; all — подряд все режимы из REWARD_MODES (отдельный каталог на каждый)",
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="DIR",
        help="Продолжить обучение: каталог trial (PPO_…) или checkpoint_… в ray_results",
    )
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=50,
        metavar="N",
        help="Остановка после N итераций (включая уже пройденные при --checkpoint)",
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
        checkpoint_path=args.checkpoint,
        training_iterations=args.training_iterations,
    )
