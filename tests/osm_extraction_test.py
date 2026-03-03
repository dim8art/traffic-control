import os
import sys

# Adding the 'src' directory to the python path to import our modules


from src.map_engine.extractor import MapExtractor

def run_extraction_test():
    """
    Test the extraction of a specific district to verify OSM connection 
    and traffic signal filtering.
    """
    # Using Lomonosov city center as a test
    target_coordinates = (59.910740, 29.776466)
    base_name = "lomonosov"
    radius = 300

    print(f"--- Starting Conversion Test: {base_name} ---")
    
    extractor = MapExtractor(location=target_coordinates, dist = radius)
    
    graph, signals = extractor.download_and_process()
    extractor.visualize_with_signals(save_path="data/network/tests/osm_extraction/preview.png")
    
    if graph is None:
        print("Test FAILED: Could not download the graph.")
        return

    num_signals = len(signals)
    print(f"Test INFO: Found {num_signals} traffic signals in the district.")
    
    adj_list = extractor.get_adjacency_list()
    if num_signals > 0:
        first_signal_id = list(adj_list.keys())[0]
        neighbors = adj_list[first_signal_id]
        print(f"Test INFO: Signal ID {first_signal_id} has {len(neighbors)} adjacent nodes.")

    filename = "test_lomonosov_network"
    output_path = extractor.save_graphml(filename)
    
    # Final verification of the file existence
    if os.path.exists(output_path):
        print(f"Test SUCCESS: Graph saved to {output_path}")
    else:
        print("Test FAILED: File was not created.")

if __name__ == "__main__":
    run_extraction_test()