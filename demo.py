import argparse
import os
import ray
import traci
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from src.simulation.env import MultiAgentTrafficEnv

def run_inference(checkpoint_path, net_file, duration, period, gui):
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
        "net_file": net_file,
        "traffic_period": period,
        "duration": duration,
        "gui": gui
    }
    env = MultiAgentTrafficEnv(env_config)

    obs, info = env.reset()
    try:
        env.runner.start(gui=gui) 
    except Exception as e:
        print(f"TraCI уже запущен или управляется средой: {e}")
    
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
    parser = argparse.ArgumentParser(description="Запуск обученного агента в SUMO")
    
    # Путь к папке чекпоинта (например, .../checkpoint_000020)
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к папке чекпоинта")
    
    # Параметры среды
    parser.add_argument("--net", type=str, required=True, help="Путь к .net.xml файлу")
    parser.add_argument("--duration", type=int, default=3600, help="Длительность (сек)")
    parser.add_argument("--period", type=float, default=0.4, help="Интенсивность трафика")
    
    # Флаг GUI (по умолчанию включен)
    parser.add_argument("--no-gui", action="store_false", dest="gui", help="Отключить графический интерфейс")
    parser.set_defaults(gui=True)

    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        net_file=args.net,
        duration=args.duration,
        period=args.period,
        gui=args.gui
    )