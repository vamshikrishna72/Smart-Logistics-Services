# Smart Logistics Optimizer

An ML-powered system for optimizing logistics operations by reducing costs and improving delivery efficiency.

## Features

- Route optimization using K-means clustering
- Delivery time prediction using Random Forest
- Cost calculation and optimization
- Interactive web interface for visualization
- Real-time route mapping

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open http://localhost:5000 in your browser

## Usage

1. Add delivery points by entering latitude, longitude, and weight
2. Click "Optimize Routes" to generate optimized delivery clusters
3. View the results and cost analysis
4. Check the interactive map visualization

## Technical Details

- Uses scikit-learn for clustering and prediction
- XGBoost for cost optimization
- Flask for the web API
- Folium for route visualization
- Geopy for distance calculations

## Future Enhancements

- Real-time traffic data integration
- Weather-based route adjustment
- Machine learning model retraining
- Mobile app integration
