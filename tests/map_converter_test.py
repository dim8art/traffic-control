import os
import sys
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.map_engine.extractor import MapExtractor
from src.map_engine.converter import MapConverter

def run_conversion_test():
    coords = (59.91074, 29.776466)
    radius = 300
    base_name = "lomonosov_mini"
    
    print(f"--- Starting Conversion Test: {base_name} ---")
    
    extractor = MapExtractor(location=coords, dist=radius)
    graph, signals = extractor.download_and_process()
    extractor.visualize_with_signals(save_path="data/network/tests/map_converter/preview.png")

    if graph is None:
        print("Step 1 FAILED: Could not download OSM data.")
        return

    osm_path = extractor.save_osm_xml(base_name)

    print("Step 2: Converting GraphML to SUMO network...")
    converter = MapConverter()
    sumo_net_path = converter.graphml_to_sumo(osm_path, base_name)
    
    if sumo_net_path and os.path.exists(sumo_net_path):
        size = os.path.getsize(sumo_net_path) / 1024
        print(f"--- Test SUCCESS ---")
        print(f"SUMO network created: {sumo_net_path}")
        print(f"File size: {size:.2f} KB")
        print(f"Traffic signals found: {len(signals)}")
    else:
        print("--- Test FAILED: SUMO file not created ---")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_conversion_test()