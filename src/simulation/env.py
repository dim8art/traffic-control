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
        # Neighbor features are intentionally down-weighted so local metrics dominate.
        self.neighbor_obs_weight = float(config.get("neighbor_obs_weight", 0.35))
        # Reward shaping weights: local component is primary, neighbor component is auxiliary.
        self.local_reward_weight = float(config.get("local_reward_weight", 1.0))
        self.neighbor_reward_weight = float(config.get("neighbor_reward_weight", 0.2))
        self.switch_penalty = float(config.get("switch_penalty", 0.1))
        self.outgoing_halt_weight = float(config.get("outgoing_halt_weight", 0.3))
        # Maximum number of decision steps without phase change for each TLS.
        self.max_idle_decision_steps = int(config.get("max_idle_decision_steps", 8))

        worker_index = config.worker_index if hasattr(config, "worker_index") else 0
        self.worker_id = f"worker_{worker_index}"
        self.worker_port = 9000 + worker_index
        
        self.runner = SumoRunner(
            self.net_file, 
            unique_id=self.worker_id,
            port=self.worker_port,
        )


        # [local_wait, local_speed, local_upstream, neigh_wait, neigh_speed, neigh_upstream]
        self.observation_space = spaces.Box(low=0, high=1000, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self._agent_ids = [] 
        self.neighbor_map = {}
        self.last_actions = {}
        self.idle_steps = {}
        self.total_forced_switches = 0
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
        self.neighbor_map = self._build_neighbor_map()
        
        observations = {a_id: self._get_local_obs(a_id) for a_id in self._agent_ids}
        self.last_actions = {a_id: 0 for a_id in self._agent_ids}
        self.idle_steps = {a_id: 0 for a_id in self._agent_ids}
        self.total_forced_switches = 0
        
        return observations, {}

    def step(self, action_dict):
        traci.switch(self.runner.unique_id)
        forced_switches_step = 0
        effective_actions = {}
        for tls_id in self._agent_ids:
            action = action_dict.get(tls_id, 0)
            if tls_id not in self._agent_ids:
                continue

            should_switch = action == 1
            if not should_switch and self.idle_steps.get(tls_id, 0) >= self.max_idle_decision_steps:
                # Hard cap on idle duration: force a phase change.
                should_switch = True
                forced_switches_step += 1

            if should_switch:
                self._handle_phase_change(tls_id)
                self.idle_steps[tls_id] = 0
                effective_actions[tls_id] = 1
            else:
                self.idle_steps[tls_id] = self.idle_steps.get(tls_id, 0) + 1
                effective_actions[tls_id] = 0

            self.last_actions[tls_id] = effective_actions[tls_id]

        for tls_id, action in action_dict.items():
            if tls_id in self._agent_ids:
                if tls_id not in effective_actions:
                    effective_actions[tls_id] = int(action == 1)

        for _ in range(15):
            traci.simulationStep()

        self.total_forced_switches += forced_switches_step
        system_stats = self.get_system_metrics()
        
        observations = {}
        rewards = {}
        for a_id in self._agent_ids:
            observations[a_id] = self._get_local_obs(a_id)
            rewards[a_id] = self._get_local_reward(a_id, effective_actions.get(a_id, 0))

        is_terminated = traci.simulation.getMinExpectedNumber() <= 0
        terminations = {a_id: is_terminated for a_id in self._agent_ids}
        terminations["__all__"] = is_terminated
        
        truncations = {a_id: False for a_id in self._agent_ids}
        truncations["__all__"] = False
        
        info = {
            "__common__": {
                "system_avg_speed": system_stats["avg_speed"],
                "system_mean_waiting_time": system_stats["mean_waiting_time"],
                "forced_switches_step": forced_switches_step,
                "forced_switches_total": self.total_forced_switches,
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
        local_waiting_cars, local_avg_speed, local_avg_upstream = self._collect_tls_metrics(tls_id)

        neighbors = self.neighbor_map.get(tls_id, [])
        if neighbors:
            neigh_wait = []
            neigh_speed = []
            neigh_upstream = []
            for neigh_id in neighbors:
                w, s, u = self._collect_tls_metrics(neigh_id)
                neigh_wait.append(w)
                neigh_speed.append(s)
                neigh_upstream.append(u)
            neighbor_wait = float(np.mean(neigh_wait))
            neighbor_speed = float(np.mean(neigh_speed))
            neighbor_upstream = float(np.mean(neigh_upstream))
        else:
            neighbor_wait = 0.0
            neighbor_speed = 0.0
            neighbor_upstream = 0.0

        weighted_neighbor_wait = neighbor_wait * self.neighbor_obs_weight
        weighted_neighbor_speed = neighbor_speed * self.neighbor_obs_weight
        weighted_neighbor_upstream = neighbor_upstream * self.neighbor_obs_weight

        return np.array(
            [
                float(local_waiting_cars),
                float(local_avg_speed),
                float(local_avg_upstream),
                float(weighted_neighbor_wait),
                float(weighted_neighbor_speed),
                float(weighted_neighbor_upstream),
            ],
            dtype=np.float32,
        )

    def _collect_tls_metrics(self, tls_id):
        waiting_cars, avg_speed = self.runner.get_tls_state(tls_id)
        lanes_in = traci.trafficlight.getControlledLanes(tls_id)
        upstream_occupancy = [traci.lane.getLastStepOccupancy(ln) for ln in lanes_in]
        avg_upstream = sum(upstream_occupancy) / len(upstream_occupancy) if upstream_occupancy else 0.0
        return float(waiting_cars), float(avg_speed), float(avg_upstream)

    def _build_neighbor_map(self):
        lane_to_tls = {}
        for tls_id in self._agent_ids:
            for lane in set(traci.trafficlight.getControlledLanes(tls_id)):
                lane_to_tls.setdefault(lane, set()).add(tls_id)

        neighbor_map = {tls_id: set() for tls_id in self._agent_ids}
        for tls_id in self._agent_ids:
            links = traci.trafficlight.getControlledLinks(tls_id)
            for link_group in links:
                for conn in link_group:
                    if len(conn) < 2:
                        continue
                    outgoing_lane = conn[1]
                    downstream_tls_ids = lane_to_tls.get(outgoing_lane, set())
                    for downstream_tls in downstream_tls_ids:
                        if downstream_tls != tls_id:
                            neighbor_map[tls_id].add(downstream_tls)
                            neighbor_map[downstream_tls].add(tls_id)

        return {k: sorted(v) for k, v in neighbor_map.items()}

    def _get_local_reward(self, tls_id, action):
        local_pressure = self._get_tls_pressure(tls_id)

        neighbors = self.neighbor_map.get(tls_id, [])
        if neighbors:
            neighbor_pressure = float(np.mean([self._get_tls_pressure(n_id) for n_id in neighbors]))
        else:
            neighbor_pressure = 0.0

        reward = -(
            self.local_reward_weight * float(local_pressure)
            + self.neighbor_reward_weight * float(neighbor_pressure)
        )

        # --- Штраф за переключение ---
        # Если действие action == 1 (смена фазы), вычитаем штраф
        if action is not None and action == 1:
            reward -= self.switch_penalty
        return reward

    def _get_tls_pressure(self, tls_id):
        # Inbound queue near current traffic light.
        lanes_in = traci.trafficlight.getControlledLanes(tls_id)
        halt_in = sum([traci.lane.getLastStepHaltingNumber(l) for l in set(lanes_in)])

        # Outbound queue that the intersection pushes traffic into.
        links = traci.trafficlight.getControlledLinks(tls_id)
        out_lanes = set()
        for link in links:
            for conn in link:
                out_lanes.add(conn[1])

        halt_out = sum([traci.lane.getLastStepHaltingNumber(l) for l in out_lanes])
        return float(halt_in + halt_out * self.outgoing_halt_weight)

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