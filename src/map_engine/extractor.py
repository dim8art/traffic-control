import osmnx as ox
import networkx as nx
import os
import logging

ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.requests_timeout = 300
ox.settings.overpass_endpoint = "https://overpass.openstreetmap.fr/api/interpreter"
ox.settings.all_oneway = True

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class MapExtractor:
    def __init__(self, location, dist=None, network_type='drive'):
        """
        Initialize the extractor.
        :param location: String representing the area (e.g., "Arbat District, Moscow") or exact coordinates (needs dist)
        :param network_type: Type of roads to download (standard is 'drive')
        """
        self.location = location
        self.dist = dist
        self.network_type = network_type
        self.graph = None
        self.signals = None

    def download_and_process(self, visualize=False):
        """
        Downloads map data from OSM and identifies traffic signal nodes.
        """
        
        try:
            if isinstance(self.location, tuple):
                logger.info(f"Downloading graph for radius: {(self.location, self.dist)}")
                if self.dist is None:
                    raise ValueError("Distance (dist) must be provided for coordinate-based search.")
                
                logger.info(f"Downloading graph around coordinates: {self.location} with radius {self.dist}m")
                self.graph = ox.graph_from_point(
                    self.location, 
                    dist=self.dist, 
                    network_type=self.network_type, 
                    simplify=False
                )
            else:
                logger.info(f"Downloading graph for place name: {self.location}")
                self.graph = ox.graph_from_place(
                    self.location, 
                    network_type=self.network_type, 
                    simplify=False
                )

        except Exception as e:
            logger.error(f"Failed to download map data: {e}")
            return None, None

        logger.info("Extracting traffic signal infrastructure...")
        
        # Convert graph nodes to a GeoDataFrame for filtering
        nodes, _ = ox.graph_to_gdfs(self.graph)
        
        # Filter nodes tagged as 'traffic_signals'
        if 'highway' in nodes.columns:
            self.signals = nodes[nodes['highway'] == 'traffic_signals']
        else:
            self.signals = []

        logger.info(f"Graph nodes: {len(self.graph.nodes)}")
        logger.info(f"Traffic signals (agents): {len(self.signals)}")
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
        if not self.graph: 
            return

        node_colors = []
        node_sizes = []
        
        for node, data in self.graph.nodes(data=True):
            is_signal = data.get('highway') == 'traffic_signals'
            
            if is_signal:
                logger.info(f"Found signal light at {data}")
                node_colors.append('#FF0000')
                node_sizes.append(80)
            else:
                node_colors.append('#555555')
                node_sizes.append(10)

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
            import matplotlib.pyplot as plt
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Signal visualization saved to {save_path}")
            plt.close(fig)
        else:
            import matplotlib.pyplot as plt
            plt.show()