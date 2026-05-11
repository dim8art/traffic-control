import os
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import traci
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from src.simulation.runner import SumoRunner


def traffic_observation_dim(
    *,
    enable_pedestrians: bool = False,
    enable_public_transport: bool = False,
) -> int:
    """Размер вектора наблюдения агента TLS: базовые 6 + по 2 признака за пешеходов и ОТ."""
    n = 6
    if enable_pedestrians:
        n += 2
    if enable_public_transport:
        n += 2
    return n


def traffic_policy_observation_space(
    *,
    enable_pedestrians: bool = False,
    enable_public_transport: bool = False,
) -> spaces.Box:
    """Пространство наблюдений для политики Ray (должно совпадать с MultiAgentTrafficEnv)."""
    dim = traffic_observation_dim(
        enable_pedestrians=enable_pedestrians,
        enable_public_transport=enable_public_transport,
    )
    return spaces.Box(low=0.0, high=1000.0, shape=(dim,), dtype=np.float32)


def traffic_env_config(
    net_file: str,
    *,
    traffic_period: float = 0.5,
    duration: int = 3600,
    gui: bool = False,
    enable_pedestrians: bool = False,
    pedestrian_period: float | None = None,
    enable_public_transport: bool = False,
    public_transport_period: float | None = None,
) -> dict:
    """
    Единый словарь для MultiAgentTrafficEnv (Ray / демо / benchmark).
    Периоды пешеходов и ОТ по умолчанию задаются в SumoRunner, если None.
    """
    cfg: dict = {
        "net_file": net_file,
        "traffic_period": traffic_period,
        "duration": duration,
        "gui": gui,
        "enable_pedestrians": enable_pedestrians,
        "enable_public_transport": enable_public_transport,
    }
    if pedestrian_period is not None:
        cfg["pedestrian_period"] = float(pedestrian_period)
    if public_transport_period is not None:
        cfg["public_transport_period"] = float(public_transport_period)
    return cfg


_TRIP_SUM_KEYS = (
    ("trip_completed_tt_sum_all", "trip_completed_count_all", "ep_tt_sum_all", "ep_tt_cnt_all"),
    ("trip_completed_tt_sum_ped", "trip_completed_count_ped", "ep_tt_sum_ped", "ep_tt_cnt_ped"),
    ("trip_completed_tt_sum_car", "trip_completed_count_car", "ep_tt_sum_car", "ep_tt_cnt_car"),
    ("trip_completed_tt_sum_pt", "trip_completed_count_pt", "ep_tt_sum_pt", "ep_tt_cnt_pt"),
)


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

            for sum_k, cnt_k, acc_s, acc_c in _TRIP_SUM_KEYS:
                ds = last_info.get(sum_k)
                dc = last_info.get(cnt_k)
                if ds is not None and dc is not None:
                    episode.user_data[acc_s] = float(episode.user_data.get(acc_s, 0.0)) + float(ds)
                    episode.user_data[acc_c] = int(episode.user_data.get(acc_c, 0)) + int(dc)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        if "waiting_times" in episode.user_data and episode.user_data["waiting_times"]:
            mean_wait = np.mean(episode.user_data["waiting_times"])
            episode.custom_metrics["system_mean_waiting_time"] = mean_wait
        
        if "avg_speeds" in episode.user_data and episode.user_data["avg_speeds"]:
            episode.custom_metrics["system_avg_speed"] = np.mean(episode.user_data["avg_speeds"])

        if int(episode.user_data.get("ep_tt_cnt_all", 0)) > 0:
            episode.custom_metrics["mean_completed_trip_time_all_s"] = (
                float(episode.user_data["ep_tt_sum_all"])
                / int(episode.user_data["ep_tt_cnt_all"])
            )
        if int(episode.user_data.get("ep_tt_cnt_ped", 0)) > 0:
            episode.custom_metrics["mean_completed_trip_time_pedestrian_s"] = (
                float(episode.user_data["ep_tt_sum_ped"])
                / int(episode.user_data["ep_tt_cnt_ped"])
            )
        if int(episode.user_data.get("ep_tt_cnt_car", 0)) > 0:
            episode.custom_metrics["mean_completed_trip_time_car_s"] = (
                float(episode.user_data["ep_tt_sum_car"])
                / int(episode.user_data["ep_tt_cnt_car"])
            )
        if int(episode.user_data.get("ep_tt_cnt_pt", 0)) > 0:
            episode.custom_metrics["mean_completed_trip_time_public_transport_s"] = (
                float(episode.user_data["ep_tt_sum_pt"])
                / int(episode.user_data["ep_tt_cnt_pt"])
            )

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

        worker_index = getattr(config, "worker_index", 0)
        self.worker_id = f"worker_{worker_index}"
        self.worker_port = 9000 + worker_index
        
        self.runner = SumoRunner(
            self.net_file, 
            unique_id=self.worker_id,
            port=self.worker_port,
        )

        self.enable_pedestrians = bool(config.get("enable_pedestrians", False))
        self.enable_public_transport = bool(config.get("enable_public_transport", False))
        self.pedestrian_period = config.get("pedestrian_period")
        self.public_transport_period = config.get("public_transport_period")

        obs_dim = traffic_observation_dim(
            enable_pedestrians=self.enable_pedestrians,
            enable_public_transport=self.enable_public_transport,
        )
        # База: local_wait, local_speed, local_upstream, neigh_*(вес w);
        # + пешеходы: число людей на контролируемых полосах (локально и у соседей×w);
        # + ОТ: остановившиеся bus на въездах (локально и у соседей×w).
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self._agent_ids = [] 
        self.neighbor_map = {}
        self.last_actions = {}
        self.idle_steps = {}
        self.total_forced_switches = 0
        self.gui_enabled = config.get("gui", False)
        self._veh_depart_times: dict[str, float] = {}
        self._ped_depart_times: dict[str, float] = {}
        super().__init__()

    def reset(self, *, seed=None, options=None, worker_port=None):
        try:
            traci.close(self.runner.unique_id)
        except Exception:
            pass

        self.runner.generate_random_traffic(
            period=self.traffic_period,
            duration=self.duration,
            enable_pedestrians=self.enable_pedestrians,
            pedestrian_period=self.pedestrian_period,
            enable_public_transport=self.enable_public_transport,
            public_transport_period=self.public_transport_period,
        )
        self.runner.create_config()
        
        self.runner.start(gui=self.gui_enabled)
        traci.switch(self.runner.unique_id)

        self._veh_depart_times = {}
        self._ped_depart_times = {}

        if not self._agent_ids:
            self._agent_ids = list(traci.trafficlight.getIDList())

        for _ in range(500):
            traci.simulationStep()
            self._travel_after_sim_step(count_completed=False)
            if traci.simulation.getMinExpectedNumber() > 0:
                break

        self._travel_align_episode_baseline()

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

        trip_batch = self._empty_trip_batch()
        for _ in range(15):
            traci.simulationStep()
            self._merge_trip_batch(trip_batch, self._travel_after_sim_step(count_completed=True))

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
                "trip_completed_tt_sum_all": trip_batch["sum_all"],
                "trip_completed_count_all": trip_batch["cnt_all"],
                "trip_completed_tt_sum_ped": trip_batch["sum_ped"],
                "trip_completed_count_ped": trip_batch["cnt_ped"],
                "trip_completed_tt_sum_car": trip_batch["sum_car"],
                "trip_completed_count_car": trip_batch["cnt_car"],
                "trip_completed_tt_sum_pt": trip_batch["sum_pt"],
                "trip_completed_count_pt": trip_batch["cnt_pt"],
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

        parts = [
            float(local_waiting_cars),
            float(local_avg_speed),
            float(local_avg_upstream),
            float(weighted_neighbor_wait),
            float(weighted_neighbor_speed),
            float(weighted_neighbor_upstream),
        ]

        if self.enable_pedestrians:
            local_ped = float(self._collect_tls_pedestrian_count(tls_id))
            if neighbors:
                neigh_ped = float(
                    np.mean(
                        [self._collect_tls_pedestrian_count(n_id) for n_id in neighbors]
                    )
                )
            else:
                neigh_ped = 0.0
            parts.append(local_ped)
            parts.append(neigh_ped * self.neighbor_obs_weight)

        if self.enable_public_transport:
            local_bus = float(self._collect_tls_halting_bus_count(tls_id))
            if neighbors:
                neigh_bus = float(
                    np.mean(
                        [self._collect_tls_halting_bus_count(n_id) for n_id in neighbors]
                    )
                )
            else:
                neigh_bus = 0.0
            parts.append(local_bus)
            parts.append(neigh_bus * self.neighbor_obs_weight)

        return np.asarray(parts, dtype=np.float32)

    def _collect_tls_metrics(self, tls_id):
        waiting_cars, avg_speed = self.runner.get_tls_state(tls_id)
        lanes_in = traci.trafficlight.getControlledLanes(tls_id)
        upstream_occupancy = [traci.lane.getLastStepOccupancy(ln) for ln in lanes_in]
        avg_upstream = sum(upstream_occupancy) / len(upstream_occupancy) if upstream_occupancy else 0.0
        return float(waiting_cars), float(avg_speed), float(avg_upstream)

    def _collect_tls_pedestrian_count(self, tls_id: str) -> int:
        """Число пешеходов на полосах, которыми управляет данный TLS (в т.ч. переходы)."""
        lanes = set(traci.trafficlight.getControlledLanes(tls_id))
        persons: set[str] = set()
        for lane_id in lanes:
            persons.update(traci.lane.getLastStepPersonIDs(lane_id))
        return len(persons)

    def _collect_tls_halting_bus_count(self, tls_id: str) -> int:
        """Остановившиеся автобусы (vclass bus) на въездных полосах перекрёстка."""
        lanes = set(traci.trafficlight.getControlledLanes(tls_id))
        halt_speed_eps = 0.1
        n = 0
        for lane_id in lanes:
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                try:
                    if traci.vehicle.getVehicleClass(vid) != "bus":
                        continue
                except traci.TraCIException:
                    continue
                try:
                    if traci.vehicle.getSpeed(vid) < halt_speed_eps:
                        n += 1
                except traci.TraCIException:
                    continue
        return n

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

    @staticmethod
    def _empty_trip_batch() -> dict[str, float | int]:
        return {
            "sum_all": 0.0,
            "cnt_all": 0,
            "sum_ped": 0.0,
            "cnt_ped": 0,
            "sum_car": 0.0,
            "cnt_car": 0,
            "sum_pt": 0.0,
            "cnt_pt": 0,
        }

    @staticmethod
    def _merge_trip_batch(dst: dict[str, float | int], src: dict[str, float | int]) -> None:
        for k in src:
            dst[k] = dst[k] + src[k]  # type: ignore[operator]

    def _travel_align_episode_baseline(self) -> None:
        """После прогрева: время в пути считаем от текущего момента для уже в сети агентов."""
        now = float(traci.simulation.getTime())
        for vid in traci.vehicle.getIDList():
            self._veh_depart_times[vid] = now
        for pid in traci.person.getIDList():
            self._ped_depart_times[pid] = now

    def _travel_after_sim_step(self, count_completed: bool) -> dict[str, float | int]:
        """
        После одного simulationStep: регистрация въездов; при count_completed — учёт завершённых поездок.
        Длительность = время прибытия − время появления в сети (или getDeparture для ТС).
        """
        batch = self._empty_trip_batch()
        now = float(traci.simulation.getTime())

        for vid in traci.simulation.getDepartedIDList():
            self._veh_depart_times[vid] = now

        for pid in traci.simulation.getDepartedPersonIDList():
            self._ped_depart_times[pid] = now

        if not count_completed:
            return batch

        for pid in traci.simulation.getArrivedPersonIDList():
            dep = self._ped_depart_times.pop(pid, None)
            if dep is None:
                continue
            tt = max(0.0, now - float(dep))
            batch["sum_ped"] += tt
            batch["cnt_ped"] += 1
            batch["sum_all"] += tt
            batch["cnt_all"] += 1

        for vid in traci.simulation.getArrivedIDList():
            dep_map = self._veh_depart_times.pop(vid, None)
            try:
                dep_sched = float(traci.vehicle.getDeparture(vid))
            except traci.TraCIException:
                dep_sched = None
            dep = dep_map if dep_map is not None else dep_sched
            if dep is None:
                continue
            tt = max(0.0, now - float(dep))
            try:
                vclass = traci.vehicle.getVehicleClass(vid)
            except traci.TraCIException:
                vclass = "passenger"
            if vclass == "bus":
                batch["sum_pt"] += tt
                batch["cnt_pt"] += 1
            else:
                batch["sum_car"] += tt
                batch["cnt_car"] += 1
            batch["sum_all"] += tt
            batch["cnt_all"] += 1

        return batch

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