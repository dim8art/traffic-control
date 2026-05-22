from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import ray
import traci
from ray.rllib.algorithms.algorithm import Algorithm

from src.rl.obs_salience import (
    format_top_saliency,
    log_saliency_tensorboard_scalar_groups,
    mean_abs_saliency_across_tls,
    observation_feature_labels,
    observation_saliency_logp_gradient,
)
from src.simulation.env import (
    REWARD_MODES,
    MultiAgentTrafficEnv,
    traffic_env_config,
    traffic_observation_dim,
)

logger = logging.getLogger(__name__)


def _obs_dim_doc(
    *,
    enable_pedestrians: bool,
    enable_public_transport: bool,
) -> str:
    dim = traffic_observation_dim(
        enable_pedestrians=enable_pedestrians,
        enable_public_transport=enable_public_transport,
    )
    return ", ".join(
        observation_feature_labels(
            dim,
            enable_pedestrians=enable_pedestrians,
            enable_public_transport=enable_public_transport,
        )
    )


def run_sumo_freerun(
    sumo_net_xml_path: str,
    duration: int,
    period: float,
    gui: bool,
    *,
    enable_pedestrians: bool = False,
    pedestrian_period: float | None = None,
    enable_public_transport: bool = False,
    public_transport_period: float | None = None,
    reward_mode: str = "pressure",
    ped_reward_weight: float | None = None,
) -> None:
    """
    SUMO со случайным трафиком без RL: фазы светофоров не переключаются (действие 0 на всех TLS).
    Используется программа регулирования из сети.
    """

    env = MultiAgentTrafficEnv(
        traffic_env_config(
            sumo_net_xml_path,
            traffic_period=period,
            duration=duration,
            gui=gui,
            enable_pedestrians=enable_pedestrians,
            pedestrian_period=pedestrian_period,
            enable_public_transport=enable_public_transport,
            public_transport_period=public_transport_period,
            reward_mode=reward_mode,
            ped_reward_weight=ped_reward_weight,
        )
    )
    obs, _info = env.reset()
    total_reward = 0.0
    logger.info("SUMO: симуляция без модели (RL отключён)")
    if gui:
        logger.info("GUI: при необходимости нажмите «Старт» в SUMO.")

    try:
        done = False
        while not done:
            actions = {aid: 0 for aid in obs}
            obs, rewards, terminated, truncated, _infos = env.step(actions)
            total_reward += float(sum(rewards.values()))
            done = terminated.get("__all__", False) or truncated.get("__all__", False)
        logger.info("Готово. Служебная сумма метрик за прогон: %.1f", total_reward)
    except Exception as e:
        logger.exception("Ошибка во время симуляции: %s", e)
    finally:
        try:
            traci.close(env.runner.unique_id)
        except Exception:
            try:
                traci.close()
            except Exception:
                pass


def run_inference(
    checkpoint_path: str,
    sumo_net_xml_path: str,
    duration: int,
    period: float,
    gui: bool,
    *,
    enable_pedestrians: bool = False,
    pedestrian_period: float | None = None,
    enable_public_transport: bool = False,
    public_transport_period: float | None = None,
    reward_mode: str = "pressure",
    ped_reward_weight: float | None = None,
    explain_obs_every: int | None = None,
    explain_tls_limit: int = 2,
    tensorboard_dir: str | None = None,
    tensorboard_saliency_every: int = 1,
) -> None:
    tb_writer = None
    try:
        traci.close()
    except Exception:
        pass

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    logger.info("Загрузка модели из: %s", checkpoint_path)
    algo = Algorithm.from_checkpoint(checkpoint_path)

    explain_policy = algo.get_policy("traffic_policy")

    tb_log_dir: str | None = None
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
        tb_writer.add_scalar("demo_saliency/_meta/writer_ready", 1.0, 0)
        tb_writer.flush()
        logger.info(
            "TensorBoard saliency: каждые %s шаг(ов) RL → %s\n"
            "  В UI: Scalars, фильтр «demo_saliency» (mean_abs/*, share_sum/*, aggregate/l1_norm).\n"
            "  Запуск: tensorboard --logdir %s\n"
            "  Saliency пишется только в demo (не в ray_results при train).",
            max(1, tensorboard_saliency_every),
            tb_log_dir,
            tb_log_dir,
        )

    # Создание среды для демонстрации
    env_config = traffic_env_config(
        sumo_net_xml_path,
        traffic_period=period,
        duration=duration,
        gui=gui,
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

    logger.info("Запуск демонстрации политики")
    logger.info("Если включён GUI, нажмите «Старт» в окне SUMO.")

    if explain_obs_every:
        logger.info(
            "Saliency (лог консоли): градиент |∂ log π(a|obs)/∂ obs| каждые %s шаг(ов); "
            "порядок компонент: %s",
            explain_obs_every,
            _obs_dim_doc(
                enable_pedestrians=enable_pedestrians,
                enable_public_transport=enable_public_transport,
            ),
        )

    rl_iteration = 0
    tb_writes = 0

    try:
        # Цикл симуляции
        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = algo.compute_single_action(
                    agent_obs,
                    policy_id="traffic_policy",
                    explore=False,
                )

            tb_every = max(1, int(tensorboard_saliency_every))
            log_tb = tb_writer is not None and rl_iteration % tb_every == 0
            log_console = bool(
                explain_obs_every and explain_obs_every > 0
            ) and (rl_iteration % explain_obs_every == 0)

            if log_console or log_tb:
                tls_ids_sorted = sorted(obs.keys())[: max(1, explain_tls_limit)]
                grad_ok: list[np.ndarray] = []
                for tls_id in tls_ids_sorted:
                    obs_vec = np.asarray(obs[tls_id], dtype=np.float32)
                    act_val = actions[tls_id]
                    action_int = int(
                        act_val.item() if hasattr(act_val, "item") else act_val,
                    )
                    try:
                        g = observation_saliency_logp_gradient(
                            explain_policy,
                            obs_vec,
                            taken_action=action_int,
                        )
                        grad_ok.append(np.asarray(g, dtype=np.float32))
                        if log_console:
                            logger.info(
                                "Saliency шаг=%s tls=%s действие=%s топ|∇| %s",
                                rl_iteration,
                                tls_id,
                                action_int,
                                format_top_saliency(
                                    g,
                                    top_k=6,
                                    enable_pedestrians=enable_pedestrians,
                                    enable_public_transport=enable_public_transport,
                                ),
                            )
                    except Exception as exc:
                        logger.warning(
                            "Saliency пропуск tls=%s: %s — нужен framework=torch "
                            "и совпадающая размерность наблюдения.",
                            tls_id,
                            exc,
                        )
                if log_tb:
                    if grad_ok:
                        aggregated = mean_abs_saliency_across_tls(grad_ok)
                        log_saliency_tensorboard_scalar_groups(
                            tb_writer,
                            rl_iteration,
                            aggregated,
                            prefix="demo_saliency",
                            enable_pedestrians=enable_pedestrians,
                            enable_public_transport=enable_public_transport,
                        )
                        tb_writes += 1
                    else:
                        logger.warning(
                            "TensorBoard: на шаге %s saliency не записана "
                            "(градиент log π не посчитан ни для одного TLS).",
                            rl_iteration,
                        )

            # Шаг среды
            obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # Считаем награду для статистики
            total_reward += sum(rewards.values())
            
            # Проверка завершения по флагу __all__
            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            rl_iteration += 1

        logger.info("Тест завершён. Суммарная награда: %s", total_reward)
        if tb_writer is not None:
            if tb_writes == 0:
                logger.warning(
                    "TensorBoard: файл событий в %s, но saliency-скаляры не записаны. "
                    "Проверьте framework=torch в чекпоинте и размерность наблюдения.",
                    tb_log_dir,
                )
            else:
                logger.info(
                    "TensorBoard: записано %s шаг(ов) saliency в %s",
                    tb_writes,
                    tb_log_dir,
                )

    except Exception as e:
        logger.exception("Ошибка во время выполнения: %s", e)
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        # Финальное закрытие соединения
        traci.close()
        ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO: демо с моделью или чистый прогон")
    parser.add_argument(
        "--sumo-only",
        action="store_true",
        help="Без RL: только сеть, трафик и программа TLS из .net.xml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Папка чекпоинта RLlib (не нужна при --sumo-only)",
    )
    
    # Параметры среды
    parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="Путь к SUMO-сети (.net.xml)",
    )
    parser.add_argument("--duration", type=int, default=3600, help="Длительность (сек)")
    parser.add_argument("--period", type=float, default=0.4, help="Интенсивность трафика")
    parser.add_argument(
        "--with-pedestrians",
        action="store_true",
        help="Добавить пешеходов (нужны пешеходные рёбра в .net.xml)",
    )
    parser.add_argument(
        "--pedestrian-period",
        type=float,
        default=None,
        metavar="SEC",
        help="Интервал появления пешеходов (по умолчанию ~2× --period)",
    )
    parser.add_argument(
        "--with-public-transport",
        action="store_true",
        help="Добавить автобусы (vclass=bus) как упрощённый ОТ",
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
        help="Функция награды среды",
    )
    parser.add_argument(
        "--ped-reward-weight",
        type=float,
        default=None,
        metavar="W",
        help="Вес пешеходов для *_sidewalk",
    )

    parser.add_argument(
        "--explain-obs-every",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="STEPS",
        help="Каждые STEPS решений RL печать топ saliency по наблюдению (градиент log π); "
        "флаг без числа ⇒ 1",
    )
    parser.add_argument(
        "--explain-tls-limit",
        type=int,
        default=2,
        help="Не больше N светофоров для saliency за шаг (консоль + TensorBoard-агрегат)",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Каталог логов TensorBoard: средние |∂log π/∂obs| по выбранным TLS (скаляры)",
    )
    parser.add_argument(
        "--tensorboard-saliency-every",
        type=int,
        default=1,
        metavar="STEPS",
        help="При --tensorboard-dir: записывать saliency раз в STEPS RL-решений",
    )

    # Флаг GUI (по умолчанию включен)
    parser.add_argument("--no-gui", action="store_false", dest="gui", help="Отключить графический интерфейс")
    parser.set_defaults(gui=True)

    args = parser.parse_args()

    from src.logging_config import configure_logging

    configure_logging()

    if args.sumo_only:
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
    else:
        if not args.checkpoint:
            parser.error("Укажите --checkpoint или запустите с --sumo-only")
        run_inference(
            checkpoint_path=args.checkpoint,
            sumo_net_xml_path=args.map,
            duration=args.duration,
            period=args.period,
            gui=args.gui,
            enable_pedestrians=args.with_pedestrians,
            pedestrian_period=args.pedestrian_period,
            enable_public_transport=args.with_public_transport,
            public_transport_period=args.public_transport_period,
            reward_mode=args.reward_mode,
            ped_reward_weight=args.ped_reward_weight,
            explain_obs_every=args.explain_obs_every,
            explain_tls_limit=args.explain_tls_limit,
            tensorboard_dir=args.tensorboard_dir,
            tensorboard_saliency_every=args.tensorboard_saliency_every,
        )