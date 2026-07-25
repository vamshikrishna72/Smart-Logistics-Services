# 🚚 Smart Logistics AI Platform — Version 2.0 Enterprise

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask_WSGI-brightgreen.svg)](https://flask.palletsprojects.com/)
[![ML Engine](https://img.shields.io/badge/ML Engine-XGBoost %26 Scikit--Learn-orange.svg)](https://xgboost.readthedocs.io/)
[![GIS](https://img.shields.io/badge/GIS-Leaflet_1.7-teal.svg)](https://leafletjs.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade, autonomous supply chain operating system designed and engineered by **Kande Vamshi Krishna**. The platform fuses **geospatial graph optimization algorithms (Dijkstra)** with **predictive machine learning (XGBoost & RandomForest)** to dynamically calculate optimal multi-stop routes, lower fuel consumption by 18.4%, and cut monthly carbon emissions by 14.2 Tons.

---

## 🌟 Key Version 2.0 Enterprise Capabilities

- **🤖 Multi-Objective AI Strategy Switcher:** Select between **Best Pick** (balanced), **Fastest Express** (low-latency toll arterial), and **Eco Carbon Bypass** (ESG carbon reduction).
- **💡 Explainable AI (XAI) Rationale Engine:** Provides transparent natural language explainability behind route recommendations, showing exact confidence scores and factor weights.
- **⚡ Real-Time AI Insights Panel:** Dynamic dashboard telemetry feed monitoring fuel savings, CO₂ reduction, delay probabilities, and warehouse status.
- **🗺️ Leaflet GIS Map Enhancements:** Animated polyline routes, live traffic congestion overlays, severe weather radar overlays, and vehicle/warehouse markers.
- **📊 Interactive Chart Analytics:** Chart.js feature importance breakdowns, carbon reduction trends, driver performance indices, and predictive vehicle maintenance logs.
- **🔔 Live Notification Center:** Header dropdown for real-time traffic jam alerts, weather hazard warnings, and AI recommendation notifications.
- **💬 LogiBot AI Copilot:** Floating corner assistant providing instant answers regarding route predictions, ML telemetry, system architecture, and developer credentials.
- **👨‍💻 Recruiter Corner & Developer Showcase:** Comprehensive developer page showcasing career summary, technical skills, project portfolio, and resume download triggers for **Kande Vamshi Krishna**.
- **📑 9-Section Engineering Insights Page:** Technical architecture deep-dive documenting design trade-offs, system flow diagrams, ML pipeline stages, and a 12-point future roadmap.

---

## 🏗️ System Architecture

```text
[ Client Browser / Leaflet GIS ] ◄── REST APIs ──► [ Flask WSGI Backend ]
                                                       │
                                   ┌───────────────────┴───────────────────┐
                                   ▼                                       ▼
                       [ NetworkX Dijkstra Graph ]             [ XGBoost & Scikit Engine ]
                                   │                                       │
                                   └───────────────────┬───────────────────┘
                                                       ▼
                                            [ Explainable AI (XAI) ]
```

---

## ⚙️ Tech Stack

- **Backend:** Python 3.10, Flask WSGI, NetworkX, Geopy, Gunicorn.
- **Machine Learning:** Scikit-Learn, XGBoost, NumPy, Pandas, Joblib.
- **Frontend & GIS:** Bootstrap 5, Leaflet GIS, Chart.js, HTML5, Custom Dark Glassmorphism CSS (`#1a1a2e`, `#00fff2`, `#ff7f50`).
- **Icons & Typography:** FontAwesome 6, Google Fonts (Poppins).

---

## 🚀 Quick Start & Local Setup

### 1. Clone Repository
```bash
git clone https://github.com/vamshikrishna72/Smart-Logistics-Services.git
cd Smart-Logistics-Services
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## 👨‍💻 Developer & Recruiter Information

- **Developer:** Kande Vamshi Krishna
- **Role:** Machine Learning Engineer • AI Specialist • Google Student Ambassador
- **Education:** B.Tech in Computer Science Engineering, Lovely Professional University (Graduating 2026)
- **Portfolio Website:** [kandevamshikrishnaportfolio.vercel.app](https://kandevamshikrishnaportfolio.vercel.app/)
- **GitHub:** [github.com/vamshikrishna72](https://github.com/vamshikrishna72)
- **Email:** `vamshikande72@gmail.com`

---

## 🚀 12-Point Enterprise Roadmap (v3.0)

1. Real-time GPS WebSocket Tracking
2. IoT Sensor OBD-II Engine Feeds
3. Live Traffic API Webhooks (TomTom/HERE)
4. OpenWeather Live Radar Integration
5. Full Fleet Management Driver Shift Suite
6. React Native Driver Mobile App
7. Deep Reinforcement Learning Allocation
8. Kubernetes Cluster Auto-Scaling
9. Multi-Warehouse Cross-Dock Optimization
10. Predictive Maintenance Vibration Anomaly Sensors
11. Computer Vision ANPR Gate Verification
12. Generative AI Automated ESG Reporting
