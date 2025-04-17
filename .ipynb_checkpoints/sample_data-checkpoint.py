import pandas as pd

# Sample logistics data
logistics_data = {
    'shipment_id': ['SH001', 'SH002', 'SH003', 'SH004', 'SH005'],
    'container_type': ['Standard', 'Refrigerated', 'Standard', 'Oversized', 'Standard'],
    'weight_kg': [1200, 800, 1500, 2000, 900],
    'volume_m3': [20, 15, 25, 35, 18],
    'origin_lat': [19.0760, 28.6139, 12.9716, 13.0827, 17.3850],
    'origin_lng': [72.8777, 77.2090, 77.5946, 80.2707, 78.4867],
    'destination_lat': [28.6139, 12.9716, 13.0827, 17.3850, 19.0760],
    'destination_lng': [77.2090, 77.5946, 80.2707, 78.4867, 72.8777],
    'delivery_priority': ['High', 'Medium', 'High', 'Low', 'Medium'],
    'transport_mode': ['Sea', 'Air', 'Truck', 'Sea', 'Truck'],
    'estimated_cost': [5000, 8000, 3000, 6000, 2500]
}

# Create DataFrame
df = pd.DataFrame(logistics_data)

# Sample container status data
container_status = {
    'container_id': ['CNT001', 'CNT002', 'CNT003', 'CNT004', 'CNT005'],
    'status': ['In Transit', 'Loading', 'Delivered', 'At Port', 'In Transit'],
    'location_lat': [22.5726, 19.0760, 13.0827, 17.3850, 25.2048],
    'location_lng': [88.3639, 72.8777, 80.2707, 78.4867, 55.2708],
    'completion_percentage': [75, 25, 100, 45, 60],
    'carrier': ['DHL', 'FedEx', 'USPS', 'Maersk', 'DHL'],
    'last_updated': ['2025-03-22 10:30', '2025-03-22 09:45', '2025-03-22 08:15', 
                    '2025-03-22 11:00', '2025-03-22 10:00']
}

# Create DataFrame
container_df = pd.DataFrame(container_status)

# Sample port locations
port_locations = {
    'port_name': ['Mumbai Port', 'Chennai Port', 'Kolkata Port', 'JNPT', 'Vizag Port'],
    'lat': [18.9256, 13.0827, 22.5726, 18.9490, 17.6868],
    'lng': [72.8777, 80.2707, 88.3639, 72.9525, 83.2185],
    'capacity_utilization': [85, 70, 75, 90, 65],
    'vessels_docked': [12, 8, 10, 15, 7]
}

# Create DataFrame
ports_df = pd.DataFrame(port_locations)

# Save to CSV files
df.to_csv('logistics_data.csv', index=False)
container_df.to_csv('container_status.csv', index=False)
ports_df.to_csv('port_locations.csv', index=False)
