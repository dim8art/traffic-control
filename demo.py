import argparse
import os
import ray
import traci
from ray.rllib.algorithms.algorithm import Algorithm
from src.simulation.env import MultiAgentTrafficEnv


def run_sumo_freerun(sumo_net_xml_path: str, duration: int, period: float, gui: bool) -> None:
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
    print("--- SUMO: симуляция без модели ---")
    if gui:
        print("GUI: при необходимости нажмите Play.")

    try:
        done = False
        while not done:
            actions = {aid: 0 for aid in obs}
            obs, rewards, terminated, truncated, _infos = env.step(actions)
            total_reward += float(sum(rewards.values()))
            done = terminated.get("__all__", False) or truncated.get("__all__", False)
        print(f"Готово. Служебная сумма метрик за прогон: {total_reward:.1f}")
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        try:
            traci.close(env.runner.unique_id)
        except Exception:
            try:
                traci.close()
            except Exception:
                pass


def run_inference(checkpoint_path, sumo_net_xml_path, duration, period, gui):
    try:
        traci.close()
    except Exception:
        pass

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print(f"Загрузка модели из: {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)

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
    
    print("--- ДЕМОНСТРАЦИЯ ЗАПУЩЕНА ---")
    print("Если выбрано GUI, нажмите кнопку 'Play' в окне SUMO.")

    try:
        # Цикл симуляции
        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = algo.compute_single_action(
                    agent_obs, 
                    policy_id="traffic_policy",
                    explore=False 
                )
            
            # Шаг среды
            obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # Считаем награду для статистики
            total_reward += sum(rewards.values())
            
            # Проверка завершения по флагу __all__
            done = terminated.get("__all__", False) or truncated.get("__all__", False)

        print(f"Тест завершен. Суммарная награда: {total_reward}")
    
    except Exception as e:
        print(f"Ошибка во время выполнения: {e}")
    finally:
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
    
    # Флаг GUI (по умолчанию включен)
    parser.add_argument("--no-gui", action="store_false", dest="gui", help="Отключить графический интерфейс")
    parser.set_defaults(gui=True)

    args = parser.parse_args()

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
        )