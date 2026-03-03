import os
import time
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import traci
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import logging

from src.simulation.runner import SumoRunner
logger = logging.getLogger(__name__)

class TrafficCallbacks(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        last_info = episode.last_info_for("__common__")
        
        if last_info:
            wait_time = last_info.get("system_mean_waiting_time")
            avg_speed = last_info.get("system_avg_speed")
            
            if wait_time is not None:
                if "waiting_times" not in episode.user_data:
                    episode.user_data["waiting_times"] = []
                episode.user_data["waiting_times"].append(wait_time)
            if avg_speed is not None:
                if "avg_speeds" not in episode.user_data:
                    episode.user_data["avg_speeds"] = []
                episode.user_data["avg_speeds"].append(avg_speed)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        if "waiting_times" in episode.user_data and episode.user_data["waiting_times"]:
            mean_wait = np.mean(episode.user_data["waiting_times"])
            episode.custom_metrics["system_mean_waiting_time"] = mean_wait
        
        if "avg_speeds" in episode.user_data and episode.user_data["avg_speeds"]:
            episode.custom_metrics["system_avg_speed"] = np.mean(episode.user_data["avg_speeds"])

class MultiAgentTrafficEnv(MultiAgentEnv):
    def __init__(self, config):
        self.net_file = config["net_file"]
        self.traffic_period = config.get("traffic_period", 0.5)
        self.duration = config.get("duration", 3600)

        worker_index = config.worker_index if hasattr(config, "worker_index") else 0
        self.worker_id = f"worker_{worker_index}"
        self.worker_port = 9000 + worker_index
        
        self.runner = SumoRunner(
            self.net_file, 
            unique_id=self.worker_id,
            port=self.worker_port,
        )


        self.observation_space = spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self._agent_ids = [] 
        self.last_actions = {}
        self.gui_enabled = config.get("gui", False)
        super().__init__()

    def reset(self, *, seed=None, options=None, worker_port=None):
        try: 
            traci.close(self.runner.unique_id)
        except: 
            pass

        self.runner.generate_random_traffic(
            period=self.traffic_period, 
            duration=self.duration
        )
        self.runner.create_config()
        
        self.runner.start(gui=self.gui_enabled)
        traci.switch(self.runner.unique_id)

        if not self._agent_ids:
            self._agent_ids = list(traci.trafficlight.getIDList())

        for _ in range(500): 
            traci.simulationStep()
            if traci.simulation.getMinExpectedNumber() > 0:
                break

        self._agent_ids = list(traci.trafficlight.getIDList())
        
        observations = {a_id: self._get_local_obs(a_id) for a_id in self._agent_ids}
        self.last_actions = {a_id: 0 for a_id in self._agent_ids}
        
        return observations, {}

    def step(self, action_dict):
        traci.switch(self.runner.unique_id)
        for tls_id, action in action_dict.items():
            if tls_id in self._agent_ids:
                if action == 1: 
                    self._handle_phase_change(tls_id)

        for _ in range(15):
            traci.simulationStep()

        system_stats = self.get_system_metrics()
        
        observations = {}
        rewards = {}
        for a_id in self._agent_ids:
            observations[a_id] = self._get_local_obs(a_id)
            rewards[a_id] = self._get_local_reward(a_id, action_dict[a_id])

        is_terminated = traci.simulation.getMinExpectedNumber() <= 0
        terminations = {a_id: is_terminated for a_id in self._agent_ids}
        terminations["__all__"] = is_terminated
        
        truncations = {a_id: False for a_id in self._agent_ids}
        truncations["__all__"] = False
        
        info = {
            "__common__": {
                "system_avg_speed": system_stats["avg_speed"],
                "system_mean_waiting_time": system_stats["mean_waiting_time"]
            }
        }
        
        return observations, rewards, terminations, truncations, info

    def _handle_phase_change(self, tls_id):
        current_phase = traci.trafficlight.getPhase(tls_id)
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        num_phases = len(logic.phases)
        next_phase = (current_phase + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)

    def _get_local_obs(self, tls_id):
        # Получаем данные о текущем TLS
        waiting_cars, avg_speed = self.runner.get_tls_state(tls_id)
        
        # Находим входящие и исходящие ребра для этого светофора
        lanes_in = traci.trafficlight.getControlledLanes(tls_id)
        
        # Получаем плотность на "подступах" (upstream)
        upstream_occupancy = [traci.lane.getLastStepOccupancy(ln) for ln in lanes_in]
        avg_upstream = sum(upstream_occupancy) / len(upstream_occupancy) if upstream_occupancy else 0

        return np.array([
            float(waiting_cars), 
            float(avg_speed),
            float(avg_upstream)
        ], dtype=np.float32)

    def _get_local_reward(self, tls_id, action):
        # --- Логика Давления (Pressure) ---
        # Входящие очереди (те, кто стоят перед красным у нас)
        lanes_in = traci.trafficlight.getControlledLanes(tls_id)
        halt_in = sum([traci.lane.getLastStepHaltingNumber(l) for l in set(lanes_in)])
        
        # Исходящие очереди (те, кто стоят у соседей впереди)
        links = traci.trafficlight.getControlledLinks(tls_id)
        out_lanes = set()
        for link in links:
            for conn in link:
                out_lanes.add(conn[1]) # ID исходящей полосы
        
        halt_out = sum([traci.lane.getLastStepHaltingNumber(l) for l in out_lanes])
        
        # Формула давления: (Входящие + Исходящие*0.3)
        # Если впереди пробка (halt_out большой), награда уменьшится
        pressure = halt_in + halt_out*0.3
        reward = -float(pressure)

        # --- Штраф за переключение ---
        # Если действие action == 1 (смена фазы), вычитаем штраф
        if action is not None and action == 1:
            reward -= 0.1 
        return reward

    def get_system_metrics(self):
        """
        Collect all traffic data
        """
        vehicle_ids = traci.vehicle.getIDList()
        if not vehicle_ids:
            return {
                "avg_speed": 0,
                "total_waiting_time": 0,
                "mean_waiting_time": 0,
                "emergency_stops": 0
            }

        total_speed = 0
        total_waiting_time = 0
        
        for v_id in vehicle_ids:
            total_speed += traci.vehicle.getSpeed(v_id)
            total_waiting_time += traci.vehicle.getWaitingTime(v_id)
            
        return {
            "avg_speed": total_speed / len(vehicle_ids),
            "total_waiting_time": total_waiting_time,
            "mean_waiting_time": total_waiting_time / len(vehicle_ids),
            "vehicle_count": len(vehicle_ids)
        }