from src.simulation.runner import SumoRunner
import os

def test_sumo_run():
    net_path = "data/sumo/lomonosov_mini.net.xml"
    
    if not os.path.exists(net_path):
        print("Network file not found! Run converter first.")
        return

    runner = SumoRunner(net_path)
    
    runner.generate_random_traffic(n_vehicles=50)
    runner.create_config()
    
    print("Starting SUMO GUI... Close the window to finish.")
    runner.run_with_monitoring(gui=True)

if __name__ == "__main__":
    test_sumo_run()