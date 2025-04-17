"""
This script contains the setup code for the Jupyter notebook.
Import this in your notebook instead of running pip directly.
"""
import sys
import subprocess

def setup_environment():
    """Install required packages using %pip"""
    required_packages = [
        'numpy==1.24.3',
        'pandas==2.0.2',
        'scikit-learn==1.2.2',
        'folium==0.14.0',
        'ipywidgets==8.0.6',
        'geopy==2.3.0',
        'ipyleaflet==0.17.2',
        'plotly==5.13.1',
        'jupyter==1.0.0',
        'notebook==6.5.4'
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package}")
    print("Setup complete!")

def setup_notebook_styling():
    """Return the custom CSS styling for the notebook"""
    return """
    <style>
        .widget-tab > .widget-tab-contents {
            padding: 20px;
        }
        .widget-tab {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px;
        }
        .jupyter-widgets-output-area {
            padding: 10px;
        }
        .widget-button {
            border-radius: 8px !important;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .widget-text, .widget-dropdown {
            border-radius: 8px !important;
        }
        .dashboard-card {
            background: #ffffff;
            border-radius: 10px;
            padding: 15px;
            margin: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #4361ee, #3f37c9);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(67, 97, 238, 0.3);
        }
    </style>
    """
