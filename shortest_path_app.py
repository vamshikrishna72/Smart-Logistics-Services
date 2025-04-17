from flask import Flask, render_template, request, jsonify
import folium
import networkx as nx
from geopy.distance import geodesic
import json

app = Flask(__name__)

class ShortestPathFinder:
    def __init__(self):
        self.graph = nx.Graph()
        self.locations = {}
        
    def add_location(self, name, lat, lon):
        """Add a location to the graph"""
        self.locations[name] = (float(lat), float(lon))
        self.graph.add_node(name, pos=(float(lat), float(lon)))
        
        # Add edges to all existing nodes
        for other_name, (other_lat, other_lon) in self.locations.items():
            if other_name != name:
                distance = geodesic((lat, lon), (other_lat, other_lon)).kilometers
                self.graph.add_edge(name, other_name, weight=distance)
                
    def find_shortest_path(self, start, end):
        """Find shortest path between two locations"""
        try:
            path = nx.shortest_path(self.graph, start, end, weight='weight')
            distance = nx.shortest_path_length(self.graph, start, end, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            return None, None
        
    def get_map_with_path(self, path=None):
        """Create a map with all locations and optional path"""
        if not self.locations:
            return None
            
        # Calculate center point
        lats = [loc[0] for loc in self.locations.values()]
        lons = [loc[1] for loc in self.locations.values()]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add markers for all locations
        for name, (lat, lon) in self.locations.items():
            folium.Marker(
                [lat, lon],
                popup=name,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
            
        # Draw path if provided
        if path and len(path) > 1:
            path_coords = []
            for point in path:
                lat, lon = self.locations[point]
                path_coords.append([lat, lon])
                
            folium.PolyLine(
                path_coords,
                weight=3,
                color='red',
                opacity=0.8
            ).add_to(m)
            
        return m

# Create global instance
path_finder = ShortestPathFinder()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('shortest_path.html')

@app.route('/add_location', methods=['POST'])
def add_location():
    """Add a new location"""
    data = request.get_json()
    name = data.get('name')
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not all([name, lat, lon]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    path_finder.add_location(name, lat, lon)
    return jsonify({
        'message': f'Added location {name}',
        'locations': list(path_finder.locations.keys())
    })

@app.route('/find_path', methods=['POST'])
def find_path():
    """Find shortest path between two points"""
    data = request.get_json()
    start = data.get('start')
    end = data.get('end')
    
    if not all([start, end]):
        return jsonify({'error': 'Missing start or end point'}), 400
        
    path, distance = path_finder.find_shortest_path(start, end)
    
    if not path:
        return jsonify({'error': 'No path found'}), 404
        
    # Create map with path
    m = path_finder.get_map_with_path(path)
    map_html = m.get_root().render()
    
    return jsonify({
        'path': path,
        'distance': distance,
        'map_html': map_html
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
