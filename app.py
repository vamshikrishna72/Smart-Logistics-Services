from flask import Flask, render_template, request, jsonify
import folium
import networkx as nx
from geopy.distance import geodesic
from route_optimizer import RouteOptimizer
import json
import joblib
import numpy as np
from datetime import datetime
from logistics_optimizer import LogisticsOptimizer, SmartLogisticsOptimizer
import pandas as pd
import logging
from load_gps_data import get_delivery_points

app = Flask(__name__)

# Initialize the route optimizer
route_optimizer = RouteOptimizer()

# Initialize the logistics optimizers
optimizer = LogisticsOptimizer()
smart_optimizer = SmartLogisticsOptimizer()

# Load the ML model
model = joblib.load('models/logistics_model.pkl')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize empty DataFrames for data that might not be available
logistics_data = pd.DataFrame()
container_status = pd.DataFrame()
port_locations = pd.DataFrame()
weather_data = pd.DataFrame()

def format_dollars(amount):
    """Format amount in US Dollars"""
    return f"${amount:.2f}"

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page"""
    return render_template('dashboard.html')

@app.route('/route_optimizer')
def route_optimizer_page():
    """Render the route optimizer page"""
    return render_template('route_optimizer.html')

@app.route('/optimizer')
def optimizer_page():
    """Render the optimizer page"""
    return render_template('index.html')

@app.route('/add_location', methods=['POST'])
def add_location():
    """Add a new location"""
    data = request.get_json()
    name = data.get('name')
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not all([name, lat, lon]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    route_optimizer.add_location(name, lat, lon)
    return jsonify({
        'message': f'Added location {name}',
        'locations': list(route_optimizer.locations.keys())
    })

@app.route('/find_path', methods=['POST'])
def find_path():
    """Find shortest path between two points"""
    try:
        data = request.get_json()
        start = data.get('start')
        end = data.get('end')
        
        if not all([start, end]):
            return jsonify({'error': 'Missing start or end point'}), 400
        
        # Build distance matrix before finding path
        route_optimizer.build_distance_matrix()
            
        path, distance = route_optimizer.find_shortest_path(start, end)
        
        if not path:
            return jsonify({'error': 'No path found'}), 404
            
        # Get coordinates for the path
        path_coords = []
        for loc_id in path:
            lat, lon = route_optimizer.locations[loc_id]
            path_coords.append(f"{lat},{lon}")
            
        # Create map with path
        m = route_optimizer.visualize_route(path)
        map_html = m.get_root().render()
        
        # Get cost prediction for this route
        weather = data.get('weather', 'clear')  # Default to clear weather
        traffic = data.get('traffic', 'low')    # Default to low traffic
        time = datetime.now().hour
        
        # Prepare features for prediction
        features = np.array([[
            distance,
            1 if traffic == 'high' else 0,      # Traffic encoding
            1 if weather == 'rain' else 0,      # Weather encoding
            time
        ]])
        
        # Get cost prediction
        predicted_cost = model.predict(features)[0]
        
        return jsonify({
            'path': path_coords,
            'distance': distance,
            'map_html': map_html,
            'predicted_cost': format_dollars(predicted_cost)
        })
    except Exception as e:
        logger.error(f"Error in find_path: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_cost', methods=['POST'])
def predict_cost():
    """Predict delivery cost based on route and conditions"""
    data = request.get_json()
    distance = data.get('distance')
    traffic = data.get('traffic', 'low')
    weather = data.get('weather', 'clear')
    time = datetime.now().hour
    
    if distance is None:
        return jsonify({'error': 'Missing distance'}), 400
    
    # Prepare features
    features = np.array([[
        distance,
        1 if traffic == 'high' else 0,
        1 if weather == 'rain' else 0,
        time
    ]])
    
    # Get prediction
    predicted_cost = model.predict(features)[0]
    
    return jsonify({
        'predicted_cost': format_dollars(predicted_cost)
    })

@app.route('/predict_route', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        
        # Extract parameters
        distance = float(data.get('distance', 0))
        traffic_level = data.get('traffic_level', 'medium')
        weather = data.get('weather', 'clear')
        vehicle_condition = data.get('vehicle_condition', 'good')
        current_hour = datetime.now().hour
        
        # Get predictions from the ML model
        duration = smart_optimizer.predict_route_duration(
            distance=distance,
            hour=current_hour,
            traffic_level=traffic_level,
            weather=weather
        )
        
        cost = smart_optimizer.predict_delivery_cost(
            distance=distance,
            duration=duration,
            traffic_level=traffic_level,
            vehicle_condition=vehicle_condition,
            weather=weather
        )
        
        # Calculate impact scores
        weather_impact = {
            'clear': 1.0,
            'rain': 1.2,
            'snow': 1.5
        }.get(weather, 1.0)
        
        traffic_impact = {
            'low': 1.0,
            'medium': 1.3,
            'high': 1.6
        }.get(traffic_level, 1.0)
        
        return jsonify({
            'predictions': {
                'estimated_duration': round(duration, 2),
                'estimated_cost': format_dollars(cost),
                'weather_impact': round(weather_impact, 2),
                'traffic_impact': round(traffic_impact, 2)
            },
            'model_info': {
                'features': {
                    'distance': 78.5,
                    'traffic': 7.9,
                    'weather': 4.3,
                    'vehicle': 2.1,
                    'time': 1.1
                },
                'accuracy': 95.2
            }
        })
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize_routes():
    try:
        data = request.get_json()
        n_vehicles = int(data.get('n_vehicles', 3))
        
        # Get delivery points from GPS data
        delivery_points = get_delivery_points()
        
        if not delivery_points:
            return jsonify({'error': 'No delivery points available'})
            
        result = optimizer.optimize_routes(delivery_points, n_vehicles)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error optimizing routes: {e}")
        return jsonify({
            'error': str(e)
        })

@app.route('/optimize_smart', methods=['POST'])
def optimize_smart():
    try:
        data = request.get_json()
        
        # Extract parameters
        hub = data.get('hub')
        delivery_points = data.get('delivery_points', [])
        n_vehicles = data.get('n_vehicles', 3)
        traffic_level = data.get('traffic_level', 'medium')
        vehicle_condition = data.get('vehicle_condition', 'good')
        weather = data.get('weather', 'clear')
        
        # Validate input
        if not hub or not delivery_points:
            return jsonify({'error': 'Missing required parameters'}), 400
            
        # Optimize routes
        result = smart_optimizer.optimize_routes(
            hub=hub,
            delivery_points=delivery_points,
            n_vehicles=n_vehicles,
            traffic_level=traffic_level,
            vehicle_condition=vehicle_condition,
            weather=weather
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error optimizing routes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/route_map')
def route_map():
    # Create a base map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    # Get delivery points from GPS data
    points = get_delivery_points()
    
    # Add markers for each point
    for point in points:
        folium.Marker(
            [point["lat"], point["lng"]],
            popup=f"{point['name']} ({point['weight']}kg)",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    return m._repr_html_()

@app.route('/api/stats')
def get_stats():
    return jsonify({
        'total_deliveries': 1234,
        'on_time_rate': 98.5,
        'avg_cost': format_dollars(12.45),
        'active_vehicles': 8,
        'total_vehicles': 10,
        'system_health': 'Optimal',
        'next_maintenance': '2 days'
    })

@app.route('/api/shipments')
def get_shipments():
    # Return sample shipment data (replace with actual data in production)
    return jsonify([])

@app.route('/api/container_status')
def get_container_status():
    # Return sample container status (replace with actual data in production)
    return jsonify({
        'total': 100,
        'in_transit': 45,
        'delivered': 50,
        'delayed': 5
    })

@app.route('/api/weather')
def get_weather():
    # Return sample weather data (replace with actual data in production)
    return jsonify({
        'condition': 'Clear',
        'temperature': 24,
        'impact': 'Low'
    })

@app.route('/api/maintenance')
def get_maintenance():
    # Return sample maintenance data (replace with actual data in production)
    return jsonify({
        'next_date': '2025-03-28',
        'vehicles_due': 2,
        'status': 'On Schedule'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
