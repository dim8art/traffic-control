import osmnx as ox
import os
import logging

from shapely import box

ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.requests_timeout = 300
ox.settings.all_oneway = True

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class MapExtractor:
    def __init__(self, 
                 location=None, 
                 dist=500, 
                 place_name=None, 
                 polygon=None, 
                 file_path=None, 
                 network_type='drive'):
        """
        Initialize the extractor with various data source options.
        """
        self.location = location      # (lat, lon) tuple
        self.dist = dist              # distance in meters for bbox
        self.place_name = place_name  # string, e.g., "Lomonosov, Saint Petersburg"
        self.polygon = polygon        # shapely.geometry.Polygon object
        self.file_path = file_path    # path to local .osm or .graphml file
        self.network_type = network_type
        
        self.graph = None
        self.signals = None

    def download_and_process(self):
        """
        Loads map data based on the provided source and truncates it to the target area.
        """
        try:
            # 1. PRIORITY: Load from local file if provided
            if self.file_path and os.path.exists(self.file_path):
                logger.info(f"Loading base graph from local file: {self.file_path}")
                if self.file_path.endswith('.osm'):
                    full_graph = ox.graph_from_xml(self.file_path, simplify=False)
                else:
                    full_graph = ox.load_graphml(self.file_path, simplify=False)
            
            # 2. OPTION: Download by place name
            elif self.place_name:
                logger.info(f"Downloading graph for place: {self.place_name}")
                full_graph = ox.graph_from_place(self.place_name, network_type=self.network_type, simplify=False)
            
            # 3. OPTION: Download by coordinates (default fallback)
            else:
                logger.info(f"Downloading graph around coordinates: {self.location}")
                # Download slightly larger area to ensure clean truncation later
                full_graph = ox.graph_from_point(self.location, dist=self.dist + 100, 
                                               network_type=self.network_type, simplify=False)

            # --- TRUNCATION LOGIC --- 
            
            # Create the bounding polygon if not explicitly provided
            target_polygon = self.polygon
            
            if target_polygon is None and self.location and self.dist:
                n, s, e, w = ox.utils_geo.bbox_from_point(self.location, dist=self.dist)
                target_polygon = box(w, s, e, n)
                logger.info(f"Created bounding box polygon for truncation (dist={self.dist}m)")

            if target_polygon:
                logger.info("Applying polygon truncation to the graph...")
                self.graph = ox.truncate.truncate_graph_polygon(
                    full_graph, 
                    target_polygon, 
                    truncate_by_edge=False
                )
            else:
                self.graph = full_graph

        except Exception as e:
            logger.error(f"Failed to process map data: {e}")
            return None, None

        # Extract traffic signals from the processed graph
        logger.info("Filtering traffic signal nodes...")
        nodes, _ = ox.graph_to_gdfs(self.graph)
        
        if 'highway' in nodes.columns:
            # 1. Standard node signals
            standard_signals = nodes[nodes['highway'] == 'traffic_signals']
            
            # 2. Add crossing signals if they exist (sometimes labeled as crossings)
            crossing_signals = nodes[nodes['highway'].isin(['crossing']) & 
                                    (nodes.get('crossing') == 'traffic_signals')]
            
            import pandas as pd
            self.signals = pd.concat([standard_signals, crossing_signals]).drop_duplicates()
        else:
            self.signals = []

        logger.info(f"Graph nodes: {len(self.graph.nodes)}")
        return self.graph, self.signals

    def get_adjacency_list(self):
        """
        Builds a neighbor map for distributed agents.
        Returns a dictionary where keys are signal IDs and values are adjacent node IDs.
        """
        adj_map = {}
        if self.signals is None:
            return adj_map
            
        signal_ids = self.signals.index.tolist()
        for node_id in signal_ids:
            neighbors = list(self.graph.neighbors(node_id))
            adj_map[node_id] = neighbors
            
        return adj_map

    def save_osm_xml(self, filename):
        """Saves the graph in native OSM XML format for SUMO compatibility."""
        output_dir = "data/network"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filepath = os.path.join(output_dir, f"{filename}.osm")
        ox.save_graph_xml(self.graph, filepath)
        logger.info(f"OSM XML saved to: {filepath}")
        return filepath

    def visualize_with_signals(self, save_path=None):
        if self.graph is None or len(self.graph.nodes) == 0:
            logger.error("Graph is empty.")
            return

        node_colors = []
        node_sizes = []
        
        signal_ids = set(self.signals.index) if hasattr(self.signals, 'index') else set()

        for node in self.graph.nodes():
            if node in signal_ids:
                node_colors.append('#FF4500') # Оранжево-красный для светофоров
                node_sizes.append(50)
            else:
                node_colors.append('#666666') # Серый для обычных узлов
                node_sizes.append(15)


        if not node_sizes:
            node_sizes = 15 # Стандартный размер, если списки пусты
            node_colors = '#666666'

        try:
            import matplotlib.pyplot as plt
            
            fig, ax = ox.plot_graph(
                self.graph,
                node_color=node_colors,
                node_size=node_sizes,
                node_alpha=0.8,
                edge_color='#CCCCCC',
                edge_linewidth=0.8,
                bgcolor='#FFFFFF',
                show=False,
                close=False
            )
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, format='svg' if save_path.endswith('.svg') else 'png')
                plt.close(fig)
                logger.info(f"Preview saved to {save_path}")
                
        except Exception as e:
            logger.error(f"Error while saving the graph {e}")