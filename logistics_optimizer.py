import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from geopy.distance import geodesic
from datetime import datetime
import folium

class SmartLogisticsOptimizer:
    def __init__(self):
        self.duration_model = self._create_duration_model()
        self.cost_model = self._create_cost_model()
        
    def _create_duration_model(self):
        """Create and train a model for predicting route duration"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Generate synthetic training data
        n_samples = 1000
        X = np.random.rand(n_samples, 4)  # distance, hour, traffic_level, weather
        
        # Simulate realistic relationships
        distances = X[:, 0] * 100  # 0-100 km
        hours = X[:, 1] * 24  # 0-24 hours
        traffic_levels = np.random.choice(['low', 'medium', 'high'], n_samples)
        weather_conditions = np.random.choice(['clear', 'rain', 'snow'], n_samples)
        
        # Convert categorical variables to numeric
        traffic_multipliers = {'low': 1.0, 'medium': 1.3, 'high': 1.6}
        weather_multipliers = {'clear': 1.0, 'rain': 1.2, 'snow': 1.5}
        
        traffic_values = np.array([traffic_multipliers[level] for level in traffic_levels])
        weather_values = np.array([weather_multipliers[condition] for condition in weather_conditions])
        
        # Calculate durations with realistic factors
        base_duration = distances * 0.02  # Base: 2 minutes per km
        time_factor = 1 + np.sin(hours * np.pi / 12) * 0.2  # ±20% variation by hour
        durations = base_duration * time_factor * traffic_values * weather_values
        
        # Train the model
        X_train = np.column_stack([distances, hours, traffic_values, weather_values])
        model.fit(X_train, durations)
        
        return model
    
    def _create_cost_model(self):
        """Create and train a model for predicting delivery costs"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Generate synthetic training data
        n_samples = 1000
        X = np.random.rand(n_samples, 5)  # distance, duration, traffic, vehicle_condition, weather
        
        # Simulate realistic relationships
        distances = X[:, 0] * 100  # 0-100 km
        durations = X[:, 1] * 5  # 0-5 hours
        traffic_levels = np.random.choice(['low', 'medium', 'high'], n_samples)
        vehicle_conditions = np.random.choice(['good', 'average', 'poor'], n_samples)
        weather_conditions = np.random.choice(['clear', 'rain', 'snow'], n_samples)
        
        # Convert categorical variables to numeric
        traffic_multipliers = {'low': 1.0, 'medium': 1.3, 'high': 1.6}
        condition_multipliers = {'good': 1.0, 'average': 1.2, 'poor': 1.5}
        weather_multipliers = {'clear': 1.0, 'rain': 1.2, 'snow': 1.5}
        
        traffic_values = np.array([traffic_multipliers[level] for level in traffic_levels])
        condition_values = np.array([condition_multipliers[cond] for cond in vehicle_conditions])
        weather_values = np.array([weather_multipliers[condition] for condition in weather_conditions])
        
        # Calculate costs with realistic factors
        base_cost = distances * 2 + durations * 30  # $2/km + $30/hour
        costs = base_cost * traffic_values * condition_values * weather_values
        
        # Train the model
        X_train = np.column_stack([distances, durations, traffic_values, condition_values, weather_values])
        model.fit(X_train, costs)
        
        return model
    
    def predict_route_duration(self, distance, hour, traffic_level='medium', weather='clear'):
        """Predict the duration of a route based on various factors"""
        # Convert categorical variables to numeric
        traffic_multipliers = {'low': 1.0, 'medium': 1.3, 'high': 1.6}
        weather_multipliers = {'clear': 1.0, 'rain': 1.2, 'snow': 1.5}
        
        traffic_value = traffic_multipliers.get(traffic_level, 1.3)
        weather_value = weather_multipliers.get(weather, 1.0)
        
        # Make prediction
        X = np.array([[distance, hour, traffic_value, weather_value]])
        duration = self.duration_model.predict(X)[0]
        
        return float(duration)
    
    def predict_delivery_cost(self, distance, duration, traffic_level='medium', 
                            vehicle_condition='good', weather='clear'):
        """Predict the cost of a delivery based on various factors"""
        # Convert categorical variables to numeric
        traffic_multipliers = {'low': 1.0, 'medium': 1.3, 'high': 1.6}
        condition_multipliers = {'good': 1.0, 'average': 1.2, 'poor': 1.5}
        weather_multipliers = {'clear': 1.0, 'rain': 1.2, 'snow': 1.5}
        
        traffic_value = traffic_multipliers.get(traffic_level, 1.3)
        condition_value = condition_multipliers.get(vehicle_condition, 1.0)
        weather_value = weather_multipliers.get(weather, 1.0)
        
        # Make prediction
        X = np.array([[distance, duration, traffic_value, condition_value, weather_value]])
        cost = self.cost_model.predict(X)[0]
        
        return float(cost)
    
    def optimize_routes(self, hub, delivery_points, n_vehicles=3, traffic_level='medium',
                       vehicle_condition='good', weather='clear'):
        """Optimize delivery routes considering ML predictions"""
        # This is a placeholder for route optimization logic
        # In a real implementation, this would use more sophisticated algorithms
        
        # Calculate basic metrics
        total_distance = 0
        total_duration = 0
        total_cost = 0
        routes = []
        
        # Simple clustering of delivery points
        points_per_vehicle = len(delivery_points) // n_vehicles
        remaining_points = len(delivery_points) % n_vehicles
        
        current_idx = 0
        for i in range(n_vehicles):
            # Assign points to this vehicle
            n_points = points_per_vehicle + (1 if i < remaining_points else 0)
            vehicle_points = delivery_points[current_idx:current_idx + n_points]
            current_idx += n_points
            
            if not vehicle_points:
                continue
            
            # Calculate route metrics
            route_distance = np.random.uniform(10, 50)  # Simplified distance calculation
            route_duration = self.predict_route_duration(
                distance=route_distance,
                hour=12,  # Assuming noon for simplicity
                traffic_level=traffic_level,
                weather=weather
            )
            
            route_cost = self.predict_delivery_cost(
                distance=route_distance,
                duration=route_duration,
                traffic_level=traffic_level,
                vehicle_condition=vehicle_condition,
                weather=weather
            )
            
            routes.append({
                'vehicle_id': i + 1,
                'points': vehicle_points,
                'distance': round(route_distance, 2),
                'duration': round(route_duration, 2),
                'cost': round(route_cost, 2)
            })
            
            total_distance += route_distance
            total_duration += route_duration
            total_cost += route_cost
        
        return {
            'routes': routes,
            'summary': {
                'total_distance': round(total_distance, 2),
                'total_duration': round(total_duration, 2),
                'total_cost': round(total_cost, 2),
                'n_vehicles': n_vehicles,
                'n_deliveries': len(delivery_points)
            }
        }

class LogisticsOptimizer:
    def __init__(self):
        # Load historical logistics data
        self.historical_data = pd.read_csv('logistics_data.csv')
        
        # Load container status data if available
        try:
            self.container_status = pd.read_csv('container_status.csv')
        except:
            self.container_status = None
            
        # Load port locations if available
        try:
            self.port_locations = pd.read_csv('port_locations.csv')
        except:
            self.port_locations = None
            
        # Initialize cost factors
        self.cost_factors = {
            'Standard': 1.0,
            'Refrigerated': 1.5,
            'Oversized': 2.0
        }
        
        self.transport_cost = {
            'Sea': 0.5,  # Cost per km per kg
            'Air': 2.0,
            'Truck': 1.0
        }

    def preprocess_data(self, data):
        """Preprocess logistics data"""
        # Handle missing values
        data = data.fillna(data.mean())
        
        # Scale numerical features
        numerical_cols = ['distance', 'weight', 'volume']
        data[numerical_cols] = StandardScaler().fit_transform(data[numerical_cols])
        return data
        
    def optimize_routes(self, delivery_points):
        if len(delivery_points) < 2:
            raise ValueError("Need at least 2 delivery points for optimization")

        # Convert delivery points to numpy array
        points = np.array([[p['lat'], p['lng']] for p in delivery_points])
        
        # Dynamically determine number of clusters based on points
        n_clusters = min(max(1, len(delivery_points) // 3), 5)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(points)
        
        # Organize points by cluster
        routes = {}
        for i in range(n_clusters):
            cluster_points = []
            cluster_indices = np.where(clusters == i)[0]
            
            for idx in cluster_indices:
                point = delivery_points[idx]
                
                # Find nearest port if available
                nearest_port = self.find_nearest_port(point['lat'], point['lng']) if self.port_locations is not None else None
                
                # Add additional information to the point
                enhanced_point = {
                    'lat': point['lat'],
                    'lng': point['lng'],
                    'weight': point['weight'],
                    'type': point['type'],
                    'nearest_port': nearest_port,
                    'estimated_delivery_time': self.estimate_delivery_time(point)
                }
                cluster_points.append(enhanced_point)
            
            # Sort points by weight for efficient loading
            cluster_points.sort(key=lambda x: x['weight'], reverse=True)
            routes[str(i)] = cluster_points

        return routes

    def calculate_route_costs(self, route_points):
        total_cost = 0
        
        if len(route_points) < 2:
            return 0
            
        for i in range(len(route_points) - 1):
            current = route_points[i]
            next_point = route_points[i + 1]
            
            # Calculate distance
            distance = geodesic(
                (current['lat'], current['lng']),
                (next_point['lat'], next_point['lng'])
            ).kilometers
            
            # Calculate base cost
            base_cost = distance * self.cost_factors[current['type']]
            
            # Add weight factor
            weight_factor = current['weight'] / 1000  # Convert to tons
            
            # Determine transport mode based on distance
            transport_mode = self.determine_transport_mode(distance)
            transport_factor = self.transport_cost[transport_mode]
            
            # Calculate segment cost
            segment_cost = base_cost * weight_factor * transport_factor
            
            # Add historical cost adjustment if available
            historical_adjustment = self.get_historical_cost_adjustment(
                current['lat'], current['lng'],
                next_point['lat'], next_point['lng']
            )
            
            segment_cost *= historical_adjustment
            total_cost += segment_cost
            
        return total_cost

    def determine_transport_mode(self, distance):
        if distance > 1000:
            return 'Air'
        elif distance > 500:
            return 'Sea'
        else:
            return 'Truck'

    def find_nearest_port(self, lat, lng):
        """Find the nearest port to a given location"""
        if self.port_locations is None:
            return None
            
        min_distance = float('inf')
        nearest_port = None
        
        try:
            for _, port in self.port_locations.iterrows():
                # Handle different possible column names for latitude/longitude
                port_lat = port.get('lat', port.get('latitude', None))
                port_lng = port.get('lng', port.get('longitude', None))
                
                if port_lat is None or port_lng is None:
                    continue
                    
                distance = geodesic((lat, lng), (port_lat, port_lng)).kilometers
                if distance < min_distance:
                    min_distance = distance
                    port_name = port.get('name', port.get('port_name', 'Unknown Port'))
                    nearest_port = {
                        'name': port_name,
                        'distance': distance,
                        'lat': port_lat,
                        'lng': port_lng
                    }
                    
            return nearest_port
        except Exception as e:
            print(f"Warning: Error finding nearest port: {e}")
            return None

    def estimate_delivery_time(self, point):
        # Base delivery time in hours
        base_time = 24
        
        # Adjust for container type
        type_factors = {
            'Standard': 1.0,
            'Refrigerated': 1.2,
            'Oversized': 1.5
        }
        
        adjusted_time = base_time * type_factors[point['type']]
        
        # Check historical data for similar deliveries
        if not self.historical_data.empty:
            similar_deliveries = self.historical_data[
                (self.historical_data['container_type'] == point['type']) &
                (abs(self.historical_data['origin_lat'] - point['lat']) < 0.1) &
                (abs(self.historical_data['origin_lng'] - point['lng']) < 0.1)
            ]
            
            if not similar_deliveries.empty:
                # Use historical average as a factor
                historical_factor = similar_deliveries['estimated_cost'].mean() / similar_deliveries['estimated_cost'].median()
                adjusted_time *= historical_factor
                
        return round(adjusted_time, 1)

    def get_historical_cost_adjustment(self, start_lat, start_lng, end_lat, end_lng):
        if self.historical_data.empty:
            return 1.0
            
        # Find similar routes in historical data
        similar_routes = self.historical_data[
            (abs(self.historical_data['origin_lat'] - start_lat) < 0.1) &
            (abs(self.historical_data['origin_lng'] - start_lng) < 0.1) &
            (abs(self.historical_data['destination_lat'] - end_lat) < 0.1) &
            (abs(self.historical_data['destination_lng'] - end_lng) < 0.1)
        ]
        
        if similar_routes.empty:
            return 1.0
            
        # Calculate adjustment factor based on historical costs
        avg_historical_cost = similar_routes['estimated_cost'].mean()
        median_historical_cost = similar_routes['estimated_cost'].median()
        
        if median_historical_cost == 0:
            return 1.0
            
        return avg_historical_cost / median_historical_cost

    def predict_delivery_time(self, route_data):
        """Predict delivery time based on route characteristics"""
        # Train model with historical data (simplified)
        X = route_data[['distance', 'traffic_score', 'num_stops']]
        y = route_data['actual_delivery_time']
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        
        return model.predict(X)

    def visualize_routes(self, routes):
        """Create a map visualization of routes"""
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # India center
        
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for cluster_id, points in routes.items():
            color = colors[int(cluster_id) % len(colors)]
            
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
        {'id': 1, 'lat': 19.0760, 'lng': 72.8777, 'weight': 10, 'type': 'Standard'},  # Mumbai
        {'id': 2, 'lat': 28.6139, 'lng': 77.2090, 'weight': 15, 'type': 'Refrigerated'},  # Delhi
        {'id': 3, 'lat': 12.9716, 'lng': 77.5946, 'weight': 8, 'type': 'Oversized'},   # Bangalore
        {'id': 4, 'lat': 17.3850, 'lng': 78.4867, 'weight': 12, 'type': 'Standard'},  # Hyderabad
        {'id': 5, 'lat': 13.0827, 'lng': 80.2707, 'weight': 9, 'type': 'Refrigerated'},   # Chennai
    ]
    
    optimizer = LogisticsOptimizer()
    optimized_routes = optimizer.optimize_routes(sample_delivery_points)
    
    # Create visualization
    route_map = optimizer.visualize_routes(optimized_routes)
    route_map.save('optimized_routes.html')
