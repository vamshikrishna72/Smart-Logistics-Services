import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from geopy.distance import geodesic
from datetime import datetime
import folium

class LogisticsOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.delivery_predictor = RandomForestRegressor(random_state=42)
        self.cost_predictor = xgb.XGBRegressor(random_state=42)
        
    def preprocess_data(self, data):
        """Preprocess logistics data"""
        # Handle missing values
        data = data.fillna(data.mean())
        
        # Scale numerical features
        numerical_cols = ['distance', 'weight', 'volume']
        data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        return data
        
    def optimize_routes(self, delivery_points):
        """Cluster delivery points for route optimization"""
        # Convert delivery points to numpy array
        points = np.array([[p['lat'], p['lng']] for p in delivery_points])
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(points)
        
        # Group delivery points by cluster
        optimized_routes = {}
        for i, cluster in enumerate(clusters):
            if cluster not in optimized_routes:
                optimized_routes[cluster] = []
            optimized_routes[cluster].append(delivery_points[i])
            
        return optimized_routes
    
    def predict_delivery_time(self, route_data):
        """Predict delivery time based on route characteristics"""
        # Train model with historical data (simplified)
        X = route_data[['distance', 'traffic_score', 'num_stops']]
        y = route_data['actual_delivery_time']
        self.delivery_predictor.fit(X, y)
        
        return self.delivery_predictor.predict(X)
    
    def calculate_route_costs(self, route):
        """Calculate costs for a given route"""
        total_distance = 0
        prev_point = None
        
        for point in route:
            if prev_point:
                distance = geodesic(
                    (prev_point['lat'], prev_point['lng']),
                    (point['lat'], point['lng'])
                ).kilometers
                total_distance += distance
            prev_point = point
            
        # Basic cost calculation (can be enhanced with more factors)
        fuel_cost_per_km = 2.5  # Example rate
        return total_distance * fuel_cost_per_km
    
    def visualize_routes(self, routes):
        """Create a map visualization of routes"""
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # India center
        
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for cluster_id, points in routes.items():
            color = colors[cluster_id % len(colors)]
            
            # Plot points and connect them with lines
            coordinates = [(p['lat'], p['lng']) for p in points]
            folium.PolyLine(coordinates, color=color, weight=2, opacity=0.8).add_to(m)
            
            # Add markers for each point
            for point in points:
                folium.Marker(
                    [point['lat'], point['lng']],
                    popup=f"Delivery Point {point.get('id', '')}"
                ).add_to(m)
                
        return m

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_delivery_points = [
        {'id': 1, 'lat': 19.0760, 'lng': 72.8777, 'weight': 10},  # Mumbai
        {'id': 2, 'lat': 28.6139, 'lng': 77.2090, 'weight': 15},  # Delhi
        {'id': 3, 'lat': 12.9716, 'lng': 77.5946, 'weight': 8},   # Bangalore
        {'id': 4, 'lat': 17.3850, 'lng': 78.4867, 'weight': 12},  # Hyderabad
        {'id': 5, 'lat': 13.0827, 'lng': 80.2707, 'weight': 9},   # Chennai
    ]
    
    optimizer = LogisticsOptimizer()
    optimized_routes = optimizer.optimize_routes(sample_delivery_points)
    
    # Create visualization
    route_map = optimizer.visualize_routes(optimized_routes)
    route_map.save('optimized_routes.html')
