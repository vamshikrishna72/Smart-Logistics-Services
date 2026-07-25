"""
Multi-Objective Machine Learning Model Trainer for Smart Logistics Platform v2.0
Engineered by Kande Vamshi Krishna
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def train_and_save_models():
    print("Training Enterprise ML Logistics Models...")
    np.random.seed(42)
    n_samples = 5000

    # 1. Feature Generation (Realistic Logistics Dataset)
    distances_km = np.random.uniform(5, 500, n_samples)          # Distance in km
    traffic_index = np.random.uniform(1.0, 2.5, n_samples)         # 1.0 Low to 2.5 High
    weather_impact = np.random.uniform(1.0, 1.8, n_samples)        # Weather hazard factor
    payload_weight_ton = np.random.uniform(1.0, 25.0, n_samples)   # Truck payload in tons
    vehicle_condition = np.random.uniform(0.8, 1.4, n_samples)     # Vehicle wear factor
    hour_of_day = np.random.randint(0, 24, n_samples)              # Time of day (0-23h)

    # Matrix X: [Distance, Traffic, Weather, Payload, VehicleWear, Hour]
    X = np.column_stack([
        distances_km,
        traffic_index,
        weather_impact,
        payload_weight_ton,
        vehicle_condition,
        hour_of_day
    ])

    # 2. Target Variables Generation (Realistic Multi-Objective Formulas with Normal Noise)
    # Target 1: Duration (ETA in minutes)
    base_speed_kmh = 55.0
    time_traffic_mult = 1.0 + 0.25 * np.sin(hour_of_day * np.pi / 12)
    eta_mins = (distances_km / base_speed_kmh * 60) * traffic_index * weather_impact * time_traffic_mult + np.random.normal(0, 3, n_samples)

    # Target 2: Fuel Consumption (Liters)
    base_fuel_rate = 0.12  # L/km
    fuel_liters = (distances_km * base_fuel_rate) * (1.0 + payload_weight_ton * 0.015) * traffic_index * vehicle_condition + np.random.normal(0, 0.5, n_samples)

    # Target 3: Carbon Emissions (kg CO2)
    co2_kg = fuel_liters * 2.68 + np.random.normal(0, 0.2, n_samples)

    # Target 4: Total Delivery Cost ($)
    cost_usd = (distances_km * 2.4) + (fuel_liters * 1.8) + (eta_mins * 0.4) + np.random.normal(0, 5, n_samples)

    # Target 5: Delay Risk (%)
    delay_risk_pct = np.clip((traffic_index - 1.0) * 35 + (weather_impact - 1.0) * 25 + np.random.normal(0, 2, n_samples), 1.0, 99.0)

    # Matrix Y: [ETA_Mins, Fuel_Liters, CO2_Kg, Cost_USD, Delay_Risk_Pct]
    Y = np.column_stack([
        np.maximum(5.0, eta_mins),
        np.maximum(0.5, fuel_liters),
        np.maximum(1.0, co2_kg),
        np.maximum(10.0, cost_usd),
        delay_risk_pct
    ])

    # 3. Model Pipeline Construction
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=35, max_depth=8, random_state=42, n_jobs=-1)))
    ])

    # Train model
    pipeline.fit(X, Y)

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Save compressed models (< 5MB for GitHub)
    joblib.dump(pipeline, 'models/multi_objective_model.pkl', compress=3)
    
    # Save single cost model for legacy compatibility
    legacy_model = RandomForestRegressor(n_estimators=25, max_depth=8, random_state=42)
    legacy_model.fit(X[:, :4], Y[:, 3]) # Features: dist, traffic, weather, payload -> Cost
    joblib.dump(legacy_model, 'models/logistics_model.pkl', compress=3)

    print("[SUCCESS] ML Models trained successfully and saved to 'models/' directory!")
    print(f"Dataset Size: {n_samples} samples | Training Features: 6 | Targets: 5 Multi-Output Metrics")

if __name__ == '__main__':
    train_and_save_models()
