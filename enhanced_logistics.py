import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from geopy.distance import geodesic
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from route_optimizer import RouteOptimizer

class EnhancedLogisticsOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.delivery_points = []
        self.route_optimizer = RouteOptimizer()  # Initialize route optimizer
        self.load_data()
        
    def load_data(self):
        """Load sample logistics data"""
        try:
            self.logistics_df = pd.read_csv('logistics_data.csv')
            self.container_df = pd.read_csv('container_status.csv')
            self.ports_df = pd.read_csv('port_locations.csv')
        except:
            print("Sample data files not found. Please run sample_data.py first.")
            
    def setup_ui(self):
        """Create modern dashboard interface"""
        # Create tabs for different sections
        self.tabs = widgets.Tab()
        
        # Dashboard tab
        dashboard_tab = self.create_dashboard_tab()
        
        # Route Optimization tab
        route_tab = self.create_route_optimization_tab()
        
        # Container Tracking tab
        tracking_tab = self.create_container_tracking_tab()
        
        # Analytics tab
        analytics_tab = self.create_analytics_tab()
        
        # Set tab contents
        self.tabs.children = [dashboard_tab, route_tab, tracking_tab, analytics_tab]
        
        # Set tab titles
        titles = ['Dashboard', 'Route Optimization', 'Container Tracking', 'Analytics']
        for i, title in enumerate(titles):
            self.tabs.set_title(i, title)
            
        display(self.tabs)
        
    def create_dashboard_tab(self):
        """Create main dashboard view"""
        # Create summary cards
        total_shipments = widgets.HTML(
            value=f'<div style="padding: 10px; background: #4361ee; color: white; border-radius: 10px; margin: 5px;">'
                  f'<h4>Total Shipments</h4><h2>{len(self.logistics_df)}</h2></div>'
        )
        
        active_containers = widgets.HTML(
            value=f'<div style="padding: 10px; background: #3f37c9; color: white; border-radius: 10px; margin: 5px;">'
                  f'<h4>Active Containers</h4><h2>{len(self.container_df[self.container_df["status"]=="In Transit"])}</h2></div>'
        )
        
        # Create map
        map_widget = self.create_dashboard_map()
        
        # Create layout
        dashboard = widgets.VBox([
            widgets.HTML('<h2>Logistics Dashboard</h2>'),
            widgets.HBox([total_shipments, active_containers]),
            map_widget
        ])
        
        return dashboard
    
    def create_route_optimization_tab(self):
        """Create route optimization interface"""
        style = {'description_width': '120px'}
        layout = widgets.Layout(width='300px')
        
        # Input fields
        self.lat_input = widgets.FloatText(description='Latitude:', style=style, layout=layout)
        self.lng_input = widgets.FloatText(description='Longitude:', style=style, layout=layout)
        self.weight_input = widgets.FloatText(description='Weight (kg):', style=style, layout=layout)
        self.type_input = widgets.Dropdown(
            description='Container Type:',
            options=['Standard', 'Refrigerated', 'Oversized'],
            style=style,
            layout=layout
        )
        
        # Add start and end point selection
        self.start_point = widgets.Dropdown(
            description='Start Point:',
            options=[],
            style=style,
            layout=layout
        )
        
        self.end_point = widgets.Dropdown(
            description='End Point:',
            options=[],
            style=style,
            layout=layout
        )
        
        # Buttons
        self.add_button = widgets.Button(
            description='Add Point',
            button_style='primary',
            icon='plus'
        )
        
        self.optimize_button = widgets.Button(
            description='Find Shortest Path',
            button_style='success',
            icon='check'
        )
        
        # Output area for map and results
        self.route_output = widgets.Output()
        
        # Button click handlers
        self.add_button.on_click(self._on_add_point)
        self.optimize_button.on_click(self._find_shortest_path)
        
        # Create layout
        inputs = widgets.VBox([
            self.lat_input, 
            self.lng_input,
            self.weight_input,
            self.type_input,
            self.start_point,
            self.end_point,
            widgets.HBox([self.add_button, self.optimize_button])
        ])
        
        return widgets.VBox([inputs, self.route_output])
        
    def _on_add_point(self, b):
        """Handle adding a new delivery point"""
        point_id = f"Point_{len(self.delivery_points) + 1}"
        lat = self.lat_input.value
        lng = self.lng_input.value
        
        if lat and lng:
            self.delivery_points.append(point_id)
            self.route_optimizer.add_location(point_id, lat, lng)
            
            # Update dropdown options
            self.start_point.options = self.delivery_points
            self.end_point.options = self.delivery_points
            
            # Clear inputs
            self.lat_input.value = None
            self.lng_input.value = None
            
            with self.route_output:
                clear_output()
                print(f"Added point {point_id} at ({lat}, {lng})")
                
    def _find_shortest_path(self, b):
        """Find and display shortest path between selected points"""
        start = self.start_point.value
        end = self.end_point.value
        
        if start and end:
            # Build distance matrix if not already built
            self.route_optimizer.build_distance_matrix()
            
            # Find shortest path
            path, distance = self.route_optimizer.find_shortest_path(start, end)
            
            if path:
                with self.route_output:
                    clear_output()
                    print(f"Shortest path: {' -> '.join(path)}")
                    print(f"Total distance: {distance:.2f} km")
                    
                    # Display route on map
                    m = self.route_optimizer.visualize_route(path)
                    display(m)
            else:
                with self.route_output:
                    clear_output()
                    print("No path found between selected points")
    
    def create_container_tracking_tab(self):
        """Create container tracking interface"""
        # Search field
        search_box = widgets.Text(
            placeholder='Search container ID...',
            layout=widgets.Layout(width='50%')
        )
        
        # Status filters
        status_filter = widgets.Dropdown(
            options=['All', 'In Transit', 'Loading', 'Delivered', 'At Port'],
            value='All',
            description='Status:',
            layout=widgets.Layout(width='200px')
        )
        
        # Container list
        container_list = widgets.Output()
        with container_list:
            display(self.container_df)
        
        # Map view
        map_view = self.create_tracking_map()
        
        return widgets.VBox([
            widgets.HBox([search_box, status_filter]),
            container_list,
            map_view
        ])
    
    def create_analytics_tab(self):
        """Create analytics and reporting interface"""
        # Create charts
        cost_chart = self.create_cost_analysis_chart()
        transport_chart = self.create_transport_mode_chart()
        
        return widgets.VBox([
            widgets.HTML('<h3>Cost Analysis</h3>'),
            cost_chart,
            widgets.HTML('<h3>Transport Mode Distribution</h3>'),
            transport_chart
        ])
    
    def create_dashboard_map(self):
        """Create interactive map for dashboard"""
        center_lat = self.logistics_df['origin_lat'].mean()
        center_lng = self.logistics_df['origin_lng'].mean()
        
        m = folium.Map(location=[center_lat, center_lng], zoom_start=5)
        
        # Add markers for active shipments
        for _, row in self.container_df[self.container_df['status'] == 'In Transit'].iterrows():
            folium.Marker(
                [row['location_lat'], row['location_lng']],
                popup=f"Container: {row['container_id']}<br>Status: {row['status']}"
            ).add_to(m)
            
        return m
    
    def create_tracking_map(self):
        """Create map for container tracking"""
        center_lat = self.container_df['location_lat'].mean()
        center_lng = self.container_df['location_lng'].mean()
        
        m = folium.Map(location=[center_lat, center_lng], zoom_start=5)
        
        # Add markers for all containers
        for _, row in self.container_df.iterrows():
            color = 'red' if row['status'] == 'In Transit' else 'blue'
            folium.Marker(
                [row['location_lat'], row['location_lng']],
                popup=f"Container: {row['container_id']}<br>Status: {row['status']}<br>Carrier: {row['carrier']}",
                icon=folium.Icon(color=color)
            ).add_to(m)
            
        return m
    
    def create_cost_analysis_chart(self):
        """Create cost analysis visualization"""
        fig = px.box(self.logistics_df, y='estimated_cost', x='transport_mode',
                    title='Cost Distribution by Transport Mode')
        return fig
    
    def create_transport_mode_chart(self):
        """Create transport mode distribution chart"""
        mode_counts = self.logistics_df['transport_mode'].value_counts()
        fig = px.pie(values=mode_counts.values, names=mode_counts.index,
                    title='Transport Mode Distribution')
        return fig
    
    def add_point(self, b):
        """Add new delivery point"""
        point = {
            'lat': self.lat_input.value,
            'lng': self.lng_input.value,
            'weight': self.weight_input.value,
            'type': self.type_input.value
        }
        
        self.delivery_points.append(point)
        
        with self.points_output:
            clear_output()
            for i, p in enumerate(self.delivery_points):
                print(f"Point {i+1}: ({p['lat']}, {p['lng']}) - {p['weight']}kg - {p['type']}")
                
        # Update map
        self.update_route_map()
    
    def optimize_routes(self, b):
        """Optimize delivery routes"""
        if len(self.delivery_points) < 2:
            with self.results_output:
                clear_output()
                print("Please add at least 2 delivery points")
            return
            
        points = np.array([[p['lat'], p['lng']] for p in self.delivery_points])
        clusters = self.kmeans.fit_predict(points)
        
        # Group points by cluster
        routes = {}
        for i, cluster in enumerate(clusters):
            if cluster not in routes:
                routes[cluster] = []
            routes[cluster].append(self.delivery_points[i])
            
        # Calculate and display results
        with self.results_output:
            clear_output()
            for cluster_id, points in routes.items():
                total_distance = self.calculate_route_distance(points)
                cost = self.calculate_route_cost(total_distance, points)
                print(f"\nRoute {cluster_id + 1}:")
                print(f"Number of stops: {len(points)}")
                print(f"Total distance: {total_distance:.2f} km")
                print(f"Estimated cost: ${cost:.2f}")
                
        # Update map with optimized routes
        self.update_route_map(routes)
    
    def calculate_route_distance(self, points):
        """Calculate total distance for a route"""
        total_distance = 0
        prev_point = None
        
        for point in points:
            if prev_point:
                distance = geodesic(
                    (prev_point['lat'], prev_point['lng']),
                    (point['lat'], point['lng'])
                ).kilometers
                total_distance += distance
            prev_point = point
            
        return total_distance
    
    def calculate_route_cost(self, distance, points):
        """Calculate route cost based on distance and cargo"""
        base_rate = 2.5  # Base rate per km
        weight_factor = sum(p['weight'] for p in points) * 0.01  # Cost factor based on weight
        return distance * (base_rate + weight_factor)
    
    def update_route_map(self, routes=None):
        """Update the route optimization map"""
        with self.map_output:
            clear_output()
            center_lat = np.mean([p['lat'] for p in self.delivery_points])
            center_lng = np.mean([p['lng'] for p in self.delivery_points])
            
            m = folium.Map(location=[center_lat, center_lng], zoom_start=5)
            
            if routes:
                colors = ['red', 'blue', 'green', 'purple', 'orange']
                for cluster_id, points in routes.items():
                    color = colors[cluster_id % len(colors)]
                    coordinates = [(p['lat'], p['lng']) for p in points]
                    
                    # Draw route line
                    folium.PolyLine(
                        coordinates,
                        color=color,
                        weight=2,
                        opacity=0.8
                    ).add_to(m)
                    
                    # Add markers
                    for i, point in enumerate(points):
                        folium.Marker(
                            [point['lat'], point['lng']],
                            popup=f"Stop {i+1}<br>Weight: {point['weight']}kg<br>Type: {point['type']}"
                        ).add_to(m)
            else:
                # Just show markers for unoptimized points
                for i, point in enumerate(self.delivery_points):
                    folium.Marker(
                        [point['lat'], point['lng']],
                        popup=f"Point {i+1}<br>Weight: {point['weight']}kg<br>Type: {point['type']}"
                    ).add_to(m)
                    
            display(m)
    
    def clear_all(self, b):
        """Clear all delivery points and results"""
        self.delivery_points = []
        
        with self.points_output:
            clear_output()
            
        with self.results_output:
            clear_output()
            
        with self.map_output:
            clear_output()
            self.update_route_map()
