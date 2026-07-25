import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import folium
from geopy.distance import geodesic

from route_optimizer import RouteOptimizer
from logistics_optimizer import LogisticsOptimizer, SmartLogisticsOptimizer
from ai_engine import ai_engine
from load_gps_data import get_delivery_points

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize route optimizers with empty locations network
route_optimizer = RouteOptimizer()
optimizer = LogisticsOptimizer()
smart_optimizer = SmartLogisticsOptimizer()

def format_dollars(amount):
    """Format amount in US Dollars"""
    return f"${amount:,.2f}"

# ==================== PAGE ROUTES ====================

@app.route('/')
def home():
    """Render the landing home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the main AI logistics dashboard"""
    return render_template('dashboard.html')

@app.route('/route_optimizer')
def route_optimizer_page():
    """Render the interactive route optimizer page"""
    return render_template('route_optimizer.html')

@app.route('/analytics')
def analytics_page():
    """Render the fleet & ML analytics page"""
    return render_template('analytics.html')

@app.route('/settings')
def settings_page():
    """Render system settings and preferences page"""
    return render_template('settings.html')

@app.route('/developer')
def developer_page():
    """Render the developer showcase & recruiter corner page"""
    return render_template('developer.html')

@app.route('/engineering_insights')
def engineering_insights_page():
    """Render the engineering insights & system architecture page"""
    return render_template('engineering_insights.html')

@app.route('/optimizer')
def optimizer_alias():
    """Alias route redirecting to index"""
    return render_template('index.html')

# ==================== ENTERPRISE AI API ENDPOINTS ====================

@app.route('/api/assistant', methods=['POST'])
def ai_assistant():
    """Process questions for LogiBot AI Assistant"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        reply = ai_engine.get_assistant_response(message)
        return jsonify({'reply': reply})
    except Exception as e:
        logger.error(f"Error in ai_assistant: {str(e)}")
        return jsonify({'reply': "LogiBot v2.0 is active. How can I assist you with route predictions or developer details?"}), 200

@app.route('/api/ai_insights', methods=['GET'])
def get_ai_insights():
    """Return real-time AI Insights telemetry feed for dashboard"""
    return jsonify(ai_engine.get_ai_insights())

@app.route('/api/predict_enterprise', methods=['POST'])
def predict_enterprise():
    """Unified enterprise AI prediction endpoint"""
    try:
        data = request.get_json() or {}
        distance = float(data.get('distance', 28.4))
        traffic = data.get('traffic', 'medium')
        weather = data.get('weather', 'clear')
        vehicle_condition = data.get('vehicle_condition', 'good')
        strategy = data.get('strategy', 'best')

        predictions = ai_engine.predict_all(
            distance_km=distance,
            traffic_level=traffic,
            weather=weather,
            vehicle_condition=vehicle_condition,
            strategy=strategy
        )
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Error in predict_enterprise: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_location', methods=['POST'])
def add_location():
    """Add a new location to RouteOptimizer graph"""
    try:
        data = request.get_json() or {}
        name = data.get('name')
        lat = data.get('lat')
        lon = data.get('lon')
        
        if not all([name, lat, lon]):
            return jsonify({'error': 'Missing required fields (name, lat, lon)'}), 400
            
        route_optimizer.add_location(name, float(lat), float(lon))
        return jsonify({
            'message': f"Added location '{name}'",
            'locations': list(route_optimizer.locations.keys())
        })
    except Exception as e:
        logger.error(f"Error adding location: {str(e)}")
        return jsonify({'error': str(e)}), 500

import re

def get_location_coords(loc_id):
    """Extract (lat, lon) tuple from location ID or string formatted coordinates"""
    if not loc_id:
        return None
    if loc_id in route_optimizer.locations:
        return route_optimizer.locations[loc_id]
    # Regex match coordinates inside string e.g. "Waypoint (17.4200, 78.4379)" or "17.42, 78.43"
    match = re.search(r'(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)', str(loc_id))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None

@app.route('/find_path', methods=['POST'])
def find_path():
    """Find shortest path between two points with multi-objective predictions"""
    try:
        data = request.get_json() or {}
        start = data.get('start')
        end = data.get('end')
        weather = data.get('weather', 'clear')
        traffic = data.get('traffic', 'low')
        strategy = data.get('strategy', 'best')
        
        if not all([start, end]):
            return jsonify({'error': 'Missing start or end point'}), 400
        
        # Extract coordinates for start and end
        s_coords = get_location_coords(start)
        e_coords = get_location_coords(end)

        if s_coords and e_coords:
            s_lat, s_lon = s_coords
            e_lat, e_lon = e_coords
            
            # Real geodesic distance in km
            real_dist = geodesic((s_lat, s_lon), (e_lat, e_lon)).kilometers
            distance = max(0.1, round(real_dist, 2))
            
            # Generate realistic curved waypoints between s_coords and e_coords
            waypoints = route_optimizer.generate_route_waypoints(s_lat, s_lon, e_lat, e_lon, num_waypoints=6)
            path_coords = [f"{wp[0]},{wp[1]}" for wp in waypoints]
            path = [start, end]
        else:
            # Graph algorithm fallback
            t_factor = {'low': 1.0, 'medium': 1.25, 'high': 1.6}.get(traffic, 1.0)
            w_factor = {'clear': 1.0, 'rain': 1.2, 'snow': 1.5}.get(weather, 1.0)
            route_optimizer.build_distance_matrix(t_factor, w_factor)
            path, distance = route_optimizer.find_shortest_path(start, end, algorithm='astar')
            path_coords = []
            for loc_id in path:
                if loc_id in route_optimizer.locations:
                    lat, lon = route_optimizer.locations[loc_id]
                    path_coords.append(f"{lat},{lon}")

        # Get enterprise predictions from ML Engine
        pred = ai_engine.predict_all(distance, traffic, weather, 'good', strategy)

        return jsonify({
            'path': path,
            'path_coords': path_coords,
            'distance': round(distance, 2),
            'predictions': pred,
            'predicted_cost': format_dollars(pred['cost_usd'])
        })
    except Exception as e:
        logger.error(f"Error in find_path: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_cost', methods=['POST'])
def predict_cost():
    """Predict delivery cost based on route parameters"""
    try:
        data = request.get_json() or {}
        distance = float(data.get('distance', 25.0))
        traffic = data.get('traffic', 'low')
        weather = data.get('weather', 'clear')
        
        pred = ai_engine.predict_all(distance, traffic, weather)
        return jsonify({
            'predicted_cost': format_dollars(pred['cost_usd']),
            'fuel_liters': pred['fuel_liters'],
            'carbon_kg': pred['carbon_kg']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_route', methods=['POST'])
def predict_route():
    """Predict route telemetry parameters"""
    try:
        data = request.get_json() or {}
        distance = float(data.get('distance', 45.0))
        traffic_level = data.get('traffic_level', 'medium')
        weather = data.get('weather', 'clear')
        vehicle_condition = data.get('vehicle_condition', 'good')
        
        pred = ai_engine.predict_all(distance, traffic_level, weather, vehicle_condition)
        
        return jsonify({
            'predictions': {
                'estimated_duration': pred['eta_minutes'],
                'estimated_cost': format_dollars(pred['cost_usd']),
                'fuel_liters': pred['fuel_liters'],
                'carbon_kg': pred['carbon_kg'],
                'weather_impact': 1.2 if weather == 'rain' else 1.0,
                'traffic_impact': 1.3 if traffic_level == 'medium' else 1.0
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
        logger.error(f"Error in predict_route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize_routes():
    try:
        data = request.get_json() or {}
        n_vehicles = int(data.get('n_vehicles', 3))
        delivery_points = get_delivery_points()
        
        if not delivery_points:
            return jsonify({'error': 'No delivery points available'})
            
        result = optimizer.optimize_routes(delivery_points, n_vehicles)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error optimizing routes: {e}")
        return jsonify({'error': str(e)})

@app.route('/optimize_smart', methods=['POST'])
def optimize_smart():
    try:
        data = request.get_json() or {}
        hub = data.get('hub')
        delivery_points = data.get('delivery_points', [])
        n_vehicles = data.get('n_vehicles', 3)
        traffic_level = data.get('traffic_level', 'medium')
        vehicle_condition = data.get('vehicle_condition', 'good')
        weather = data.get('weather', 'clear')
        
        if not hub or not delivery_points:
            return jsonify({'error': 'Missing required parameters'}), 400
            
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
    """Render base Leaflet HTML map centered on region"""
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    points = get_delivery_points()
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
        'next_maintenance': '2 days',
        'ml_accuracy': '95.2%',
        'carbon_reduced': '14.2 Tons'
    })

@app.route('/api/shipments')
def get_shipments():
    return jsonify([
        {'id': 'SHP-9041', 'origin': 'Central Hub', 'destination': 'Secunderabad Depot', 'status': 'In Transit', 'eta': '24 mins'},
        {'id': 'SHP-9042', 'origin': 'HiTech Depot', 'destination': 'Outer Ring Express', 'status': 'Delivered', 'eta': 'On Time'},
        {'id': 'SHP-9043', 'origin': 'Outer Ring Expressway', 'destination': 'Central Hub', 'status': 'In Transit', 'eta': '18 mins'}
    ])

@app.route('/api/container_status')
def get_container_status():
    return jsonify({
        'total': 100,
        'in_transit': 45,
        'delivered': 50,
        'delayed': 5
    })

@app.route('/api/weather')
def get_weather():
    return jsonify({
        'condition': 'Clear',
        'temperature': 24,
        'impact': 'Low Risk',
        'humidity': '45%'
    })

@app.route('/api/maintenance')
def get_maintenance():
    return jsonify({
        'next_date': '2026-07-28',
        'vehicles_due': 2,
        'status': 'On Schedule',
        'health_index': '98.4%'
    })

@app.route('/api/assistant', methods=['POST'])
def api_assistant():
    """AI Assistant Chatbot Endpoint"""
    try:
        data = request.get_json() or {}
        user_msg = data.get('message', '')
        reply = ai_engine.get_assistant_response(user_msg)
        return jsonify({'reply': reply, 'response': reply})
    except Exception as e:
        logger.error(f"Error in assistant API: {e}")
        return jsonify({
            'reply': "Sorry, an error occurred processing your query.",
            'response': "Sorry, an error occurred processing your query."
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
