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
    observation_saliency_logp_gradient,
)
from src.simulation.env import MultiAgentTrafficEnv

logger = logging.getLogger(__name__)


OBS_DIM_DOC = ", ".join(
    (
        "local_waiting",
        "local_speed",
        "local_upstream_occ",
        "neighbor_waiting×w",
        "neighbor_speed×w",
        "neighbor_upstream×w",
    )
)


def run_sumo_freerun(
    sumo_net_xml_path: str,
    duration: int,
    period: float,
    gui: bool,
) -> None:
    """
    SUMO со случайным трафиком без RL: фазы светофоров не переключаются (действие 0 на всех TLS).
    Используется программа регулирования из сети.
    """

    env = MultiAgentTrafficEnv(
        {
            "net_file": sumo_net_xml_path,
            "traffic_period": period,
            "duration": duration,
            "gui": gui,
        }
    )
    obs, _info = env.reset()
    total_reward = 0.0
    logger.info("SUMO: симуляция без модели (RL отключён)")
    if gui:
        logger.info("GUI: при необходимости нажмите «Play» в SUMO.")

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

    if tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "Для --tensorboard-dir установите зависимость: pip install tensorboard"
            ) from exc
        os.makedirs(tensorboard_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(
            "TensorBoard saliency: каждые %s шаг(ов) RL → %s  (Scalars: "
            "demo_saliency/mean_abs/*, demo_saliency/share_sum/*). "
            "Просмотр: tensorboard --logdir %s",
            max(1, tensorboard_saliency_every),
            tensorboard_dir,
            tensorboard_dir,
        )

    # 4. Создание среды для демонстрации
    env_config = {
        "net_file": sumo_net_xml_path,
        "traffic_period": period,
        "duration": duration,
        "gui": gui
    }
    env = MultiAgentTrafficEnv(env_config)

    obs, info = env.reset()

    done = False
    total_reward = 0

    logger.info("Запуск демонстрации политики")
    logger.info("Если включён GUI, нажмите «Play» в окне SUMO.")

    if explain_obs_every:
        logger.info(
            "Saliency (лог консоли): градиент |∂ log π(a|obs)/∂ obs| каждые %s шаг(ов); "
            "порядок компонент: %s",
            explain_obs_every,
            OBS_DIM_DOC,
        )

    rl_iteration = 0

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
                                "Saliency step=%s tls=%s action=%s top|∇| %s",
                                rl_iteration,
                                tls_id,
                                action_int,
                                format_top_saliency(g, top_k=6),
                            )
                    except Exception as exc:
                        logger.warning(
                            "Saliency пропуск tls=%s: %s — нужен framework=torch "
                            "и совпадающая размерность наблюдения.",
                            tls_id,
                            exc,
                        )
                if log_tb and grad_ok:
                    aggregated = mean_abs_saliency_across_tls(grad_ok)
                    log_saliency_tensorboard_scalar_groups(
                        tb_writer,
                        rl_iteration,
                        aggregated,
                        prefix="demo_saliency",
                    )

            # Шаг среды
            obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # Считаем награду для статистики
            total_reward += sum(rewards.values())
            
            # Проверка завершения по флагу __all__
            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            rl_iteration += 1

        logger.info("Тест завершён. Суммарная награда: %s", total_reward)

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
        "--explain-obs-every",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="STEPS",
        help="Каждые STEPS решений RL печать top saliency по наблюдению (градиент log π); "
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
        help="Каталог логов TensorBoard: средние |∂log π/∂obs| по выбранным TLS (Scalars)",
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
        run_sumo_freerun(args.map, args.duration, args.period, args.gui)
    else:
        if not args.checkpoint:
            parser.error("Укажите --checkpoint или запустите с --sumo-only")
        run_inference(
            checkpoint_path=args.checkpoint,
            sumo_net_xml_path=args.map,
            duration=args.duration,
            period=args.period,
            gui=args.gui,
            explain_obs_every=args.explain_obs_every,
            explain_tls_limit=args.explain_tls_limit,
            tensorboard_dir=args.tensorboard_dir,
            tensorboard_saliency_every=args.tensorboard_saliency_every,
        )