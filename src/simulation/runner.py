from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import traci
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    logging.getLogger(__name__).error(
        "Нужна переменная окружения SUMO_HOME (корень установки SUMO).",
    )
    sys.exit(1)

logger = logging.getLogger(__name__)


class SumoRunner:
    def __init__(
        self,
        net_file: str,
        output_dir: str = "data/sumo",
        unique_id: str = "default",
        port: int = 8813,
    ) -> None:
        # Создаем персональную подпапку для воркера, чтобы файлы не перемешивались
        self.worker_dir = os.path.join(os.path.abspath(output_dir), f"run_{unique_id}")
        os.makedirs(self.worker_dir, exist_ok=True)
        
        self.net_file = os.path.join(self.worker_dir, "map.net.xml")
        if not os.path.exists(self.net_file):
            shutil.copy(net_file, self.net_file)
        self.unique_id = unique_id
            
        self.cfg_file = os.path.join(self.worker_dir, "sim.sumocfg")
        self.rou_file = os.path.join(self.worker_dir, "routes.rou.xml")
        self.ped_rou_file = os.path.join(self.worker_dir, "pedestrians.rou.xml")
        self.pt_rou_file = os.path.join(self.worker_dir, "public_transport.rou.xml")
        self.port = port
        self.simulation_duration = 3600
        self._include_pedestrians = False
        self._include_public_transport = False

    def _build_random_trips_cmd(
        self,
        out_routes: str,
        period: float,
        duration: int,
        *,
        vclass: str | None = "passenger",
        pedestrians: bool = False,
    ) -> list[str]:
        random_trips = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")
        cmd: list[str] = [
            "python3",
            random_trips,
            "-n",
            self.net_file,
            "-p",
            str(period),
            "-e",
            str(duration),
            "-o",
            out_routes,
            "--route-file",
            out_routes,
            "--validate",
            "--fringe-factor",
            "10",
            "--remove-loops",
            "--random",
            "-t",
            'departLane="best" departSpeed="0" departPos="base"',
        ]
        if pedestrians:
            cmd.append("--pedestrians")
        else:
            cmd.extend(["--vclass", vclass or "passenger"])
        return cmd

    def _run_random_trips(
        self, cmd: list[str], *, label: str, empty_fallback_path: str
    ) -> bool:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("SUMO %s: маршруты записаны", label)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(
                "randomTrips (%s) не удался: %s",
                label,
                (e.stderr or str(e)).strip(),
            )
            with open(empty_fallback_path, "w", encoding="utf-8") as f:
                f.write("<routes></routes>")
            return False

    def generate_random_traffic(
        self,
        period=0.5,
        duration=3600,
        *,
        enable_pedestrians=False,
        pedestrian_period=None,
        enable_public_transport=False,
        public_transport_period=None,
    ):
        """
        Генерация маршрутов для легкового трафика; опционально пешеходы (--pedestrians)
        и наземный ОТ (класс vclass=bus). Сеть должна содержать подходящие рёбра
        (тротуары для людей); иначе randomTrips может завершиться с ошибкой —
        тогда соответствующий файл будет пустым и не подключается к sumocfg.
        """
        self.simulation_duration = int(duration)
        self._include_pedestrians = False
        self._include_public_transport = False

        veh_cmd = self._build_random_trips_cmd(
            self.rou_file, period, duration, pedestrians=False, vclass="passenger"
        )
        if not self._run_random_trips(veh_cmd, label="легковой трафик", empty_fallback_path=self.rou_file):
            logger.info("Маршруты легкового транспорта: пустой файл (fallback)")

        if enable_pedestrians:
            p_period = (
                float(pedestrian_period)
                if pedestrian_period is not None
                else max(period * 2.0, 1.0)
            )
            ped_cmd = self._build_random_trips_cmd(
                self.ped_rou_file,
                p_period,
                duration,
                pedestrians=True,
            )
            if self._run_random_trips(
                ped_cmd, label="пешеходы", empty_fallback_path=self.ped_rou_file
            ):
                self._include_pedestrians = True

        if enable_public_transport:
            pt_period = (
                float(public_transport_period)
                if public_transport_period is not None
                else max(period * 3.0, 2.0)
            )
            pt_cmd = self._build_random_trips_cmd(
                self.pt_rou_file,
                pt_period,
                duration,
                pedestrians=False,
                vclass="bus",
            )
            if self._run_random_trips(
                pt_cmd, label="общественный транспорт (bus)", empty_fallback_path=self.pt_rou_file
            ):
                self._include_public_transport = True

    def create_config(self):
        route_parts = [os.path.abspath(self.rou_file)]
        if self._include_pedestrians:
            route_parts.append(os.path.abspath(self.ped_rou_file))
        if self._include_public_transport:
            route_parts.append(os.path.abspath(self.pt_rou_file))
        route_files_value = ",".join(route_parts)
        end_t = int(self.simulation_duration)
        # Используем абсолютные пути, чтобы SUMO точно нашел файлы
        config_content = f"""
        <configuration>
            <input>
                <net-file value="{os.path.abspath(self.net_file)}"/>
                <route-files value="{route_files_value}"/>
            </input>
            <time>
                <begin value="0"/>
                <end value="{end_t}"/>
                <step-length value="1.0"/>
            </time>
            <processing>
                <time-to-teleport value="300"/>
                <max-num-vehicles value="2000"/>
                <no-warnings value="true"/>
            </processing>
        </configuration>
        """
        with open(self.cfg_file, "w") as f:
            f.write(config_content)

    def start(self, gui=False):
        """Метод для вызова из env.reset()"""
        sumo_binary = "sumo-gui" if gui else "sumo"

        cmd = [
            sumo_binary, 
            "-c", self.cfg_file,
            "--no-warnings", "true",
            "--quit-on-end", "true",
            "--start", "true" 
        ]
        
        traci.start(cmd, port=self.port, label=self.unique_id, numRetries=10)

    def get_tls_state(self, tls_id):
        # Твой код без изменений, он корректен
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(controlled_lanes))
        total_waiting_cars = 0
        total_speed = 0.0
        for lane_id in unique_lanes:
            total_waiting_cars += traci.lane.getLastStepHaltingNumber(lane_id)
            total_speed += traci.lane.getLastStepMeanSpeed(lane_id)
        avg_speed = total_speed / len(unique_lanes) if unique_lanes else 0.0
        return total_waiting_cars, avg_speed

    def run_with_monitoring(self, gui=True, log_traffic=True):
        """
        Run simulation with traffic monitoring
        """
        sumo_binary = "sumo-gui" if gui else "sumo"
        traci.start([sumo_binary, "-c", self.cfg_file])
        
        tls_ids = traci.trafficlight.getIDList()
        logger.info("Светофоры по ID: %s", tls_ids)

        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            if step % 20 == 0:
                for tls_id in tls_ids:
                    waiting_count, avg_speed = self.get_tls_state(tls_id)
                    
                    if waiting_count > 0:
                        if log_traffic:
                            logger.info(
                                "Шаг %s | TLS %s | очередь: %s ТС | Vср=%.2f м/с",
                                step,
                                tls_id,
                                waiting_count,
                                avg_speed,
                            )
            
            step += 1
            if step > 1000: 
                break
            
        traci.close()