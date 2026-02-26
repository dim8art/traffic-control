import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import traci
from ray.rllib.algorithms.callbacks import DefaultCallbacks

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
        self.runner = config["runner"]
        
        import traci
        traci.start(["sumo", "-n", config["net_file"], "--no-warnings", "true"])
        self._agent_ids = set(traci.trafficlight.getIDList())
        traci.close()

        self.observation_space = spaces.Dict({
            a_id: spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32)
            for a_id in self._agent_ids
        })
        self.action_space = spaces.Dict({
            a_id: spaces.Discrete(2)
            for a_id in self._agent_ids
        })
        
        super().__init__()

    def reset(self, *, seed=None, options=None):
        try: 
            traci.close()
        except: 
            pass
        
        self.runner.generate_random_traffic(n_vehicles=100)
        self.runner.create_config()
        
        traci.start(["sumo", "-c", self.runner.cfg_file, "--no-warnings", "true"])
        
        self._agent_ids = set(traci.trafficlight.getIDList())
        
        observations = {a_id: self._get_local_obs(a_id) for a_id in self._agent_ids}
        return observations, {}

    def step(self, action_dict):
        for tls_id, action in action_dict.items():
            if tls_id in self._agent_ids:
                if action == 1:
                    self._handle_phase_change(tls_id)

        for _ in range(5):
            traci.simulationStep()

        system_stats = self.get_system_metrics()

        observations = {a_id: self._get_local_obs(a_id) for a_id in self._agent_ids}
        rewards = {a_id: self._get_local_reward(a_id) for a_id in self._agent_ids}
        
        # RLlib ожидает флаг завершения для всей среды под ключом "__all__"
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
        waiting_cars, _ = self.runner.get_tls_state(tls_id)
        return np.array([float(waiting_cars)], dtype=np.float32)

    def _get_local_reward(self, tls_id):
        waiting_cars, _ = self.runner.get_tls_state(tls_id)
        return -float(waiting_cars)
    
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