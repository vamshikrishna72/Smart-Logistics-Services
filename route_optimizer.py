import networkx as nx
import numpy as np
from geopy.distance import geodesic
import folium
from typing import List, Tuple, Dict

class RouteOptimizer:
    def __init__(self):
        self.graph = nx.Graph()
        self.locations = {}
        self.distance_matrix = None
        
    def add_location(self, location_id: str, lat: float, lon: float) -> None:
        """Add a location to the route network"""
        self.locations[location_id] = (lat, lon)
        self.graph.add_node(location_id, pos=(lat, lon))
        
    def build_distance_matrix(self) -> None:
        """Build distance matrix between all locations"""
        location_ids = list(self.locations.keys())
        n = len(location_ids)
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                loc1 = self.locations[location_ids[i]]
                loc2 = self.locations[location_ids[j]]
                distance = geodesic(loc1, loc2).kilometers
                
                # Add edge to graph with distance as weight
                self.graph.add_edge(location_ids[i], location_ids[j], weight=distance)
                
                # Update distance matrix
                self.distance_matrix[i, j] = distance
                self.distance_matrix[j, i] = distance
                
    def find_shortest_path(self, start_id: str, end_id: str) -> Tuple[List[str], float]:
        """Find shortest path between two locations using Dijkstra's algorithm"""
        if start_id not in self.locations or end_id not in self.locations:
            raise ValueError("Start or end location not found")
            
        try:
            path = nx.shortest_path(self.graph, start_id, end_id, weight='weight')
            distance = nx.shortest_path_length(self.graph, start_id, end_id, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            return None, None
            
    def visualize_route(self, path: List[str] = None) -> folium.Map:
        """Visualize the route on an interactive map"""
        if not self.locations:
            raise ValueError("No locations added to visualize")
            
        # Calculate center point for the map
        lats = [loc[0] for loc in self.locations.values()]
        lons = [loc[1] for loc in self.locations.values()]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add markers for all locations
        for loc_id, (lat, lon) in self.locations.items():
            folium.Marker(
                [lat, lon],
                popup=loc_id,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
        
        # Draw path if provided
        if path and len(path) > 1:
            path_coords = [[self.locations[loc_id][0], self.locations[loc_id][1]] 
                         for loc_id in path]
            folium.PolyLine(
                path_coords,
                weight=3,
                color='red',
                opacity=0.8
            ).add_to(m)
            
        return m

    def optimize_multiple_stops(self, stops: List[str]) -> Tuple[List[str], float]:
        """Find optimal route through multiple stops using nearest neighbor algorithm"""
        if len(stops) < 2:
            return stops, 0
            
        unvisited = stops[1:]
        current = stops[0]
        route = [current]
        total_distance = 0
        
        while unvisited:
            # Find nearest unvisited location
            nearest = min(unvisited, 
                        key=lambda x: nx.shortest_path_length(self.graph, current, x, weight='weight'))
            path, distance = self.find_shortest_path(current, nearest)
            
            route.extend(path[1:])  # Exclude first point as it's already in route
            total_distance += distance
            
            current = nearest
            unvisited.remove(nearest)
            
        return route, total_distance
