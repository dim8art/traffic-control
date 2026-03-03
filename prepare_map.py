import os
import sys
import argparse
import osmnx as ox
from shapely.geometry import Polygon

# Adding 'src' to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.map_engine.extractor import MapExtractor
from src.map_engine.converter import MapConverter

KRESTOVSKY_POLY = (
    "30.200,59.970;30.210,59.983;30.245,59.985;30.275,59.980;30.280,59.975;30.275,59.967;30.250,59.962;30.220,59.962;30.200,59.970"
)
KOLOMNA_POLY = (
    "30.288,59.935;30.305,59.932;30.308,59.925;30.295,59.918;30.285,59.919;30.275,59.928;30.288,59.935"
)

# Available Overpass mirrors
overpass_endpoints = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
]

ox.settings.overpass_endpoint = overpass_endpoints[2]

def main(lat, lon, radius, name, osm_data_path, place_name, polygon_coords):
    # Output paths
    osm_xml_path = f"data/network/{name}.osm"
    
    os.makedirs("data/network", exist_ok=True)
    os.makedirs("data/sumo", exist_ok=True)

    # Convert polygon coordinates if provided (expected format: lon1,lat1;lon2,lat2;...)
    target_polygon = None
    if polygon_coords:
        try:
            points = [tuple(map(float, p.split(','))) for p in polygon_coords.split(';')]
            target_polygon = Polygon(points)
            print(f"Using custom polygon with {len(points)} vertices")
        except Exception as e:
            print(f"Error parsing polygon: {e}. Falling back to radius/location.")

    print(f"--- Processing Location: {name} ---")
    
    # Initialize Extractor with all available options
    extractor = MapExtractor(
        location=(lat, lon), 
        dist=radius, 
        file_path=osm_data_path, # Path to local .osm/.pbf
        place_name=place_name,   # e.g. "Lomonosov, Russia"
        polygon=target_polygon   # shapely Polygon
    )
    
    graph, signals = extractor.download_and_process()
    
    if graph is None or len(graph.nodes) == 0:
        print("Error: Could not obtain map data. Check your coordinates, file path or connection.")
        return

    # Visualization for debugging
    extractor.visualize_with_signals(save_path=f"data/network/{name}_preview.svg")
    
    # Save the processed graph to OSM XML for SUMO converter
    extractor.save_osm_xml(name)
    print(f"OSM XML saved. Found {len(signals)} traffic signals (TLS).")

    print(f"Converting to SUMO format...")
    converter = MapConverter()
    
    # Note: Ensure your MapConverter.osm_to_sumo matches this signature
    success = converter.osm_to_sumo(input_path=osm_xml_path, output_filename=name)
    
    if success:
        print(f"Success: Network '{name}' is ready for training/testing.")
    else:
        print("Error: SUMO conversion failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Map Preparation Pipeline")
    
    parser.add_argument("--name", type=str, default="default_location", help="Name for output files")
    parser.add_argument("--lat", type=float, default=None, help="Center latitude")
    parser.add_argument("--lon", type=float, default=None, help="Center longitude")
    parser.add_argument("--radius", type=int, default=500, help="Truncation radius (square box)")
    parser.add_argument("--file", type=str, default=None, help="Path to local .osm or .pbf file")
    parser.add_argument("--place", type=str, default=None, help="Place name (e.g. 'Lomonosov, Russia')")
    
    parser.add_argument("--polygon", type=str, default=KOLOMNA_POLY, 
                        help="Polygon vertices: 'lon1,lat1;lon2,lat2;lon3,lat3'")

    args = parser.parse_args()
    
    main(
        lat=args.lat, 
        lon=args.lon, 
        radius=args.radius, 
        name=args.name, 
        osm_data_path=args.file,
        place_name=args.place,
        polygon_coords=args.polygon
    )