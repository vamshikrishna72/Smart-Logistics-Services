import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Generate synthetic training data
np.random.seed(42)
n_samples = 1000

# Features
distances = np.random.uniform(10, 500, n_samples)  # distances in km
traffic_levels = np.random.choice([0, 1], n_samples)  # 0: low, 1: high
weather_conditions = np.random.choice([0, 1], n_samples)  # 0: clear, 1: rain
times = np.random.randint(0, 24, n_samples)  # hour of day

# Combine features
X = np.column_stack([distances, traffic_levels, weather_conditions, times])

# Generate target variable (cost) with some realistic patterns
base_cost = distances * 2  # Base cost proportional to distance
traffic_impact = traffic_levels * distances * 0.3  # 30% increase in high traffic
weather_impact = weather_conditions * distances * 0.2  # 20% increase in bad weather
time_impact = np.sin(times * 2 * np.pi / 24) * distances * 0.1  # Time of day impact
noise = np.random.normal(0, 10, n_samples)  # Random variations

y = base_cost + traffic_impact + weather_impact + time_impact + noise

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'models/logistics_model.pkl')
