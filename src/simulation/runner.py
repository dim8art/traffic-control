import os
import shutil
import subprocess
import sys
import time
import traci
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

class SumoRunner:
    def __init__(self, net_file, output_dir="data/sumo", unique_id="default", port=8813):
        # Создаем персональную подпапку для воркера, чтобы файлы не перемешивались
        self.worker_dir = os.path.join(os.path.abspath(output_dir), f"run_{unique_id}")
        os.makedirs(self.worker_dir, exist_ok=True)
        
        self.net_file = os.path.join(self.worker_dir, "map.net.xml")
        if not os.path.exists(self.net_file):
            shutil.copy(net_file, self.net_file)
        self.unique_id = unique_id
            
        self.cfg_file = os.path.join(self.worker_dir, "sim.sumocfg")
        self.rou_file = os.path.join(self.worker_dir, "routes.rou.xml")
        self.port = port

    def generate_random_traffic(self, period=0.5, duration=3600):   
        random_trips = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
        
        cmd = [
            "python3", random_trips,
            "-n", self.net_file,
            "-p", str(period),
            "-e", str(duration),
            "-o", self.rou_file,
            "--route-file", self.rou_file,
            "--validate",
            "--fringe-factor", "10",
            "--remove-loops",
            "--vclass", "passenger",
            "--random",
            "-t", 'departLane="best" departSpeed="0" departPos="base"',
        ]
        try:
            # Используем shell=False для безопасности
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Traffic generated: {self.rou_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Duarouter failed! {e.stderr}")
            with open(self.rou_file, "w") as f:
                f.write('<routes></routes>')

    def create_config(self):
        # Используем абсолютные пути, чтобы SUMO точно нашел файлы
        config_content = f"""
        <configuration>
            <input>
                <net-file value="{os.path.abspath(self.net_file)}"/>
                <route-files value="{os.path.abspath(self.rou_file)}"/>
            </input>
            <time>
                <begin value="0"/>
                <end value="3600"/>
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
        logger.info(f"All traffic lights available: {tls_ids}")

        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            if step % 20 == 0:
                for tls_id in tls_ids:
                    waiting_count, avg_speed = self.get_tls_state(tls_id)
                    
                    if waiting_count > 0:
                        if log_traffic:
                            logger.info(
                                f"Step {step} | Tls {tls_id}: "
                                f"queue: {waiting_count} cars, "
                                f"avg speed: {avg_speed:.2f} m/s"
                            )
            
            step += 1
            if step > 1000: 
                break
            
        traci.close()