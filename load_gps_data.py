import pandas as pd
import numpy as np

def load_gps_data():
    """Load and process GPS data for delivery points"""
    # Sample GPS data - replace with your actual data loading logic
    data = {
        'name': ['Point A', 'Point B', 'Point C', 'Point D', 'Point E'],
        'lat': [19.0760, 18.5204, 17.3850, 18.9490, 19.2183],
        'lng': [72.8777, 73.8567, 78.4867, 72.9525, 72.9781],
        'weight': [100, 150, 200, 175, 125]  # in kg
    }
    return pd.DataFrame(data)

def get_delivery_points():
    """Get delivery points in the format expected by the optimizer"""
    df = load_gps_data()
    delivery_points = []
    
    for _, row in df.iterrows():
        point = {
            'name': row['name'],
            'lat': float(row['lat']),
            'lng': float(row['lng']),
            'weight': float(row['weight'])
        }
        delivery_points.append(point)
    
    return delivery_points
