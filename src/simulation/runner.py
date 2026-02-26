import os
import sys
import traci
import logging

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

logger = logging.getLogger(__name__)

class SumoRunner:
    def __init__(self, net_file, output_dir="data/sumo"):
        self.net_file = net_file
        self.output_dir = output_dir
        self.cfg_file = os.path.join(output_dir, "simulation.sumocfg")
        self.rou_file = os.path.join(output_dir, "simulation.rou.xml")

    def generate_random_traffic(self, n_vehicles=100):
        """Generate random trips for n cars"""
        logger.info(f"Generating random traffic: {n_vehicles} vehicles")

        random_trips = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
        
        cmd = [
            "python3", random_trips,
            "-n", self.net_file,
            "-e", str(n_vehicles),
            "-o", self.rou_file,
            "--route-file", self.rou_file
        ]
        import subprocess
        subprocess.run(cmd, check=True)

    def create_config(self):
        """Create .sumocfg file."""
        config_content = f"""
        <configuration>
            <input>
                <net-file value="{os.path.basename(self.net_file)}"/>
                <route-files value="{os.path.basename(self.rou_file)}"/>
            </input>
            <time>
                <begin value="0"/>
                <end value="3600"/>
            </time>
        </configuration>
        """
        with open(self.cfg_file, "w") as f:
            f.write(config_content)
        logger.info(f"Config created: {self.cfg_file}")

    
    def get_tls_state(self, tls_id):
        """
        Collect traffic data from tls.
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        unique_lanes = list(set(controlled_lanes))
        
        total_waiting_cars = 0
        for lane_id in unique_lanes:
            # Считаем машины, скорость которых < 0.1 м/с (стоят в пробке)
            total_waiting_cars += traci.lane.getLastStepHaltingNumber(lane_id)
            
        return total_waiting_cars, unique_lanes

    def run_with_monitoring(self, gui=True, log_traffic=True):
        """
        Run simulation with traffic monitoring
        """
        sumo_binary = "sumo-gui" if gui else "sumo"
        traci.start([sumo_binary, "-c", self.cfg_file])
        
        tls_ids = traci.trafficlight.getIDList()
        logger.info(f"All traffic lights available: {tls_ids}")
        logger.info(f"Monitoring traffic lights: {tls_ids}")

        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            if step % 20 == 0:
                for tls_id in tls_ids:
                    waiting_count, lanes = self.get_tls_state(tls_id)
                    if waiting_count > 0:
                        if log_traffic:
                            logger.info(f"Step {step} | Tls {tls_id}: queue with {waiting_count} cars")
            
            step += 1
            if step > 1000: 
                break
            
        traci.close()