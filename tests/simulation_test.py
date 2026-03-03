import sys
from src.simulation.runner import SumoRunner
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_sumo_run():
    net_path = "data/sumo/kolomna.net.xml"
    
    if not os.path.exists(net_path):
        print("Network file not found! Run converter first.")
        return

    runner = SumoRunner(net_path)
    
    runner.generate_random_traffic()
    runner.create_config()
    
    print("Starting SUMO GUI... Close the window to finish.")
    runner.run_with_monitoring(gui=True)

if __name__ == "__main__":
    test_sumo_run()