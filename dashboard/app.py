"""
ClimateGuard AI v3.0 - Interactive Dashboard
Streamlit application for climate risk intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import pickle

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_integration.openmeteo import OpenMeteoAPI, OpenMeteoAirQualityAPI
from src.api_integration.geocoding import GeocodingAPI

# Model loading
@st.cache_resource
def load_models():
    """Load trained ML models"""
    models = {}
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    try:
        # Load Climate Risk Scorer
        with open(os.path.join(model_dir, '09_climate_risk_scorer.pkl'), 'rb') as f:
            models['risk_scorer'] = pickle.load(f)
        st.success("Climate Risk Scorer loaded successfully")
    except FileNotFoundError:
        st.warning("Climate Risk Scorer model not found. Run notebook 09 first.")
    
    try:
        # Load K-Means Clustering Model
        with open(os.path.join(model_dir, '06_kmeans_clustering_model.pkl'), 'rb') as f:
            models['kmeans'] = pickle.load(f)
        st.success("K-Means clustering model loaded successfully")
    except FileNotFoundError:
        st.warning("K-Means model not found. Run notebook 06 first.")
    
    try:
        # Load Anomaly Detector
        with open(os.path.join(model_dir, '10_anomaly_detector.pkl'), 'rb') as f:
            models['anomaly_detector'] = pickle.load(f)
        st.success("Anomaly Detector loaded successfully")
    except FileNotFoundError:
        st.warning("Anomaly Detector model not found. Run notebook 10 first.")
    
    return models

# Page configuration
st.set_page_config(
    page_title="ClimateGuard AI v3.0",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'aqi_data' not in st.session_state:
    st.session_state.aqi_data = None
if 'location' not in st.session_state:
    st.session_state.location = None
if 'models' not in st.session_state:
    st.session_state.models = load_models()


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌤️ ClimateGuard AI v3.0</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Powered Climate Risk Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            [
                "Home",
                "Live Weather",
                "Historical Data",
                "Rainfall Prediction",
                "Heatwave Prediction",
                "Climate Risk Score",
                "Climate Profiling",
                "Anomaly Detection",
                "Air Quality",
                "Weather Classification",
                "Recommendations"
            ]
        )
        
        st.markdown("---")
        st.header("Location Settings")
        
        # City input
        city = st.text_input("Enter City Name", value="Delhi")
        country = st.text_input("Country (Optional)", value="India")
        
        # Manual coordinates
        st.subheader("Or Enter Coordinates")
        lat = st.number_input("Latitude", value=28.6139, format="%.4f")
        lon = st.number_input("Longitude", value=77.2090, format="%.4f")
        
        # Fetch weather button
        if st.button("Fetch Live Weather", type="primary"):
            fetch_weather_data(city, country, lat, lon)
        
        st.markdown("---")
        st.header("About")
        st.info("""
        **ClimateGuard AI v3.0**
        
        An end-to-end AI platform for climate risk intelligence.
        
        Features:
        - Real-time weather data
        - Climate risk prediction
        - Air quality monitoring
        - Weather recommendations
        """)
    
    # Page content
    if page == "Home":
        home_page()
    elif page == "Live Weather":
        live_weather_page()
    elif page == "Historical Data":
        historical_data_page()
    elif page == "Rainfall Prediction":
        rainfall_prediction_page()
    elif page == "Heatwave Prediction":
        heatwave_prediction_page()
    elif page == "Climate Risk Score":
        climate_risk_page()
    elif page == "Climate Profiling":
        climate_profiling_page()
    elif page == "Anomaly Detection":
        anomaly_detection_page()
    elif page == "Air Quality":
        air_quality_page()
    elif page == "Weather Classification":
        weather_classification_page()
    elif page == "Recommendations":
        recommendations_page()


def fetch_weather_data(city: str, country: str, lat: float, lon: float):
    """Fetch weather and air quality data"""
    
    with st.spinner("Fetching weather data..."):
        # Geocoding
        geocoding = GeocodingAPI()
        if city:
            coords = geocoding.get_coordinates(city, country)
            if coords:
                lat = coords['latitude']
                lon = coords['longitude']
                st.session_state.location = coords
                st.success(f"Found location: {coords['name']}, {coords['country']}")
            else:
                st.warning(f"Could not find coordinates for {city}. Using manual coordinates.")
        
        # Fetch weather
        weather_api = OpenMeteoAPI()
        weather_data = weather_api.get_current_weather(lat, lon, hourly=True, daily=True)
        st.session_state.weather_data = weather_data
        
        # Fetch air quality
        aqi_api = OpenMeteoAirQualityAPI()
        aqi_data = aqi_api.get_air_quality(lat, lon)
        st.session_state.aqi_data = aqi_data
        
        st.success("Weather data fetched successfully!")


def home_page():
    """Home page with overview"""
    
    st.header("🏠 Welcome to ClimateGuard AI")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Features", "22", "Modules")
        st.metric("Data Sources", "3", "APIs + Historical")
    
    with col2:
        st.metric("ML Models", "8", "Algorithms")
        st.metric("Predictions", "5", "Types")
    
    with col3:
        st.metric("Visualizations", "15", "Charts")
        st.metric("API Integrations", "3", "Services")
    
    st.markdown("---")
    
    st.header("📊 Project Overview")
    
    st.markdown("""
    ### ClimateGuard AI v3.0 is an end-to-end AI platform that:
    
    - **Analyzes** historical weather data
    - **Integrates** real-time weather information from Open-Meteo API
    - **Predicts** climate risks (rainfall, heatwaves, air quality)
    - **Detects** weather anomalies
    - **Explains** AI predictions using SHAP values
    - **Provides** actionable recommendations
    
    ### Key Features:
    
    1. **Real-Time Weather Integration** - Live weather data from Open-Meteo API
    2. **Climate Risk Scoring** - Comprehensive risk assessment (0-100)
    3. **Air Quality Monitoring** - PM2.5, PM10, and AQI predictions
    4. **Weather Classification** - Multi-class weather condition prediction
    5. **Explainable AI** - SHAP-based model explanations
    6. **Recommendation Engine** - Personalized weather recommendations
    """)
    
    st.markdown("---")
    
    st.header("🚀 Quick Start")
    
    st.markdown("""
    1. **Enter a city name** in the sidebar (e.g., Delhi, Mumbai, Bangalore)
    2. **Click "Fetch Live Weather"** to get real-time data
    3. **Navigate to different pages** to explore features:
       - Live Weather: Current conditions and forecasts
       - Historical Data: Explore historical weather patterns
       - Rainfall Prediction: Predict rainfall probability
       - Heatwave Prediction: Assess heatwave risk
       - Climate Risk Score: Overall climate risk assessment
       - Air Quality: Monitor air quality index
       - Weather Classification: Classify weather conditions
       - Recommendations: Get personalized weather advice
    """)


def live_weather_page():
    """Live weather page"""
    
    st.header("🌡️ Live Weather")
    
    if st.session_state.weather_data is None:
        st.warning("Please fetch weather data from the sidebar first.")
        return
    
    weather = st.session_state.weather_data
    
    # Current weather
    if 'current' in weather:
        current = weather['current']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Temperature", f"{current.get('temperature_2m', 0):.1f}°C")
        
        with col2:
            st.metric("Humidity", f"{current.get('relative_humidity_2m', 0):.0f}%")
        
        with col3:
            st.metric("Wind Speed", f"{current.get('wind_speed_10m', 0):.1f} m/s")
        
        with col4:
            st.metric("Pressure", f"{current.get('pressure_msl', 0):.0f} hPa")
    
    st.markdown("---")
    
    # Hourly forecast
    if 'hourly' in weather:
        st.subheader("Hourly Forecast")
        
        hourly = weather['hourly']
        time_data = hourly.get('time', [])
        temp_data = hourly.get('temperature_2m', [])
        
        if time_data and temp_data:
            df_hourly = pd.DataFrame({
                'Time': pd.to_datetime(time_data),
                'Temperature': temp_data
            })
            
            fig = px.line(df_hourly, x='Time', y='Temperature', 
                         title='Temperature Forecast (24h)')
            fig.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Daily forecast
    if 'daily' in weather:
        st.subheader("Daily Forecast")
        
        daily = weather['daily']
        dates = daily.get('time', [])
        temp_max = daily.get('temperature_2m_max', [])
        temp_min = daily.get('temperature_2m_min', [])
        precip = daily.get('precipitation_sum', [])
        
        if dates and temp_max:
            df_daily = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Max Temp': temp_max,
                'Min Temp': temp_min,
                'Precipitation': precip
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_daily['Date'], y=df_daily['Max Temp'],
                                     name='Max Temp', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df_daily['Date'], y=df_daily['Min Temp'],
                                     name='Min Temp', line=dict(color='blue')))
            
            fig.update_layout(title='Daily Temperature Forecast',
                            xaxis_title="Date", yaxis_title="Temperature (°C)")
            st.plotly_chart(fig, use_container_width=True)


def historical_data_page():
    """Historical data page"""
    
    st.header("📈 Historical Data Explorer")
    
    st.info("Historical data analysis requires the IndianWeatherRepository.csv file.")
    
    st.markdown("""
    ### Features:
    
    - Load historical weather dataset
    - Explore historical patterns
    - Analyze trends over time
    - Compare different regions
    
    ### Coming Soon:
    - Interactive data explorer
    - Time series analysis
    - Regional comparisons
    """)


def rainfall_prediction_page():
    """Rainfall prediction page"""
    
    st.header("🌧️ Rainfall Prediction")
    
    st.info("Rainfall prediction module will be available after training ML models.")
    
    st.markdown("""
    ### Features:
    
    - Rain / No Rain classification
    - Rainfall probability
    - Rainfall intensity prediction
    - Heavy rain alerts
    
    ### Models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    """)


def heatwave_prediction_page():
    """Heatwave prediction page"""
    
    st.header("🔥 Heatwave Prediction")
    
    st.info("Heatwave prediction module will be available after training ML models.")
    
    st.markdown("""
    ### Features:
    
    - Heatwave risk assessment
    - Heatwave probability
    - Temperature trend analysis
    - Heat alerts
    
    ### Models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    """)


def climate_risk_page():
    """Climate risk score page"""
    
    st.header("⚠️ Climate Risk Score")
    
    models = st.session_state.get('models', {})
    risk_scorer = models.get('risk_scorer')
    
    if risk_scorer is None:
        st.warning("Climate Risk Scorer model not loaded. Please run notebook 09 first.")
        st.markdown("""
        ### Risk Components:
        
        - Temperature
        - Humidity
        - Rainfall
        - Air Quality
        - UV Index
        - Wind Speed
        - Pressure
        
        ### Risk Levels:
        - Low (0-25)
        - Moderate (26-50)
        - High (51-75)
        - Extreme (76-100)
        """)
        return
    
    if st.session_state.weather_data is None:
        st.warning("Please fetch weather data from the sidebar first.")
        return
    
    weather = st.session_state.weather_data
    aqi = st.session_state.aqi_data
    
    # Extract features from weather data
    current = weather.get('current', {})
    
    temperature = current.get('temperature_2m', 0)
    humidity = current.get('relative_humidity_2m', 0)
    rainfall = current.get('precipitation', 0)
    wind_speed = current.get('wind_speed_10m', 0)
    pressure = current.get('pressure_msl', 0)
    uv_index = current.get('uv_index', 0)
    
    # Extract air quality data
    pm25 = 0
    if aqi and 'current' in aqi:
        pm25 = aqi['current'].get('pm2_5', 0)
    
    # Calculate risk score
    try:
        risk = risk_scorer.calculate_overall_risk(
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
            pm25=pm25,
            uv_index=uv_index,
            wind_speed=wind_speed,
            pressure=pressure
        )
        
        # Display risk score
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Risk Score", f"{risk['overall_score']:.1f}/100")
        
        with col2:
            st.metric("Risk Category", risk['category'])
        
        with col3:
            st.metric("Risk Level", risk['color'])
        
        st.markdown("---")
        
        # Display component scores
        st.subheader("Component Risk Scores")
        component_df = pd.DataFrame.from_dict(risk['component_scores'], orient='index', columns=['Score'])
        component_df = component_df.sort_values('Score', ascending=False)
        
        fig = px.bar(component_df, x=component_df.index, y='Score', 
                     title='Component Risk Scores',
                     labels={'index': 'Component', 'Score': 'Risk Score'},
                     color='Score',
                     color_continuous_scale='RdYlGn_r')
        fig.update_layout(xaxis_title="Component", yaxis_title="Risk Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk recommendations
        st.markdown("---")
        st.subheader("Risk-Based Recommendations")
        
        if risk['overall_score'] < 25:
            st.success("✅ **Low Risk**: Conditions are favorable. Normal activities can continue.")
        elif risk['overall_score'] < 50:
            st.info("⚠️ **Moderate Risk**: Some caution advised. Monitor conditions.")
        elif risk['overall_score'] < 75:
            st.warning("🔶 **High Risk**: Take precautions. Limit outdoor activities if sensitive.")
        else:
            st.error("🔴 **Extreme Risk**: Severe conditions. Avoid outdoor activities, follow official alerts.")
        
    except Exception as e:
        st.error(f"Error calculating risk score: {str(e)}")


def climate_profiling_page():
    """Climate profiling page using K-Means clustering"""
    
    st.header("🌍 Climate Profiling")
    
    models = st.session_state.get('models', {})
    kmeans = models.get('kmeans')
    
    if kmeans is None:
        st.warning("K-Means clustering model not loaded. Please run notebook 06 first.")
        st.markdown("""
        ### Climate Profiling Features:
        
        - Climate zone classification using K-Means clustering
        - Optimal number of clusters determined by silhouette analysis
        - Cluster characteristics analysis
        - Climate zone distribution visualization
        
        ### Clustering Algorithms:
        - K-Means (Primary)
        - DBSCAN
        - Gaussian Mixture
        - Agglomerative Clustering
        """)
        return
    
    if st.session_state.weather_data is None:
        st.warning("Please fetch weather data from the sidebar first.")
        return
    
    weather = st.session_state.weather_data
    
    # Extract features from weather data
    current = weather.get('current', {})
    
    # Create feature vector for prediction
    # Note: This is a simplified version. In production, you'd need to match
    # the exact features used during training
    try:
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Load the scaler if available
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        try:
            with open(os.path.join(model_dir, '06_clustering_scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)
        except FileNotFoundError:
            scaler = StandardScaler()
            st.warning("Scaler not found, using StandardScaler")
        
        # Extract relevant features (adjust based on your training data)
        features = np.array([[
            current.get('temperature_2m', 0),
            current.get('relative_humidity_2m', 0),
            current.get('wind_speed_10m', 0),
            current.get('pressure_msl', 0),
            current.get('precipitation', 0),
            current.get('uv_index', 0)
        ]])
        
        # Scale features
        features_scaled = scaler.fit_transform(features)
        
        # Predict climate zone
        climate_zone = kmeans.predict(features_scaled)[0]
        
        # Display climate zone
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Climate Zone", f"Zone {climate_zone}")
        
        with col2:
            zone_names = {0: "Temperate", 1: "Tropical"}
            st.metric("Zone Type", zone_names.get(climate_zone, "Unknown"))
        
        st.markdown("---")
        
        # Climate zone characteristics
        st.subheader("Climate Zone Characteristics")
        
        zone_info = {
            0: {
                "name": "Temperate Zone",
                "description": "Moderate temperatures, seasonal variations",
                "typical_conditions": "Mild summers, cool winters"
            },
            1: {
                "name": "Tropical Zone",
                "description": "Warm temperatures, high humidity",
                "typical_conditions": "Hot and humid year-round"
            }
        }
        
        info = zone_info.get(climate_zone, {})
        
        if info:
            st.info(f"**{info['name']}**\n\n{info['description']}\n\nTypical conditions: {info['typical_conditions']}")
        
        # Display current conditions
        st.markdown("---")
        st.subheader("Current Weather Conditions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Temperature", f"{current.get('temperature_2m', 0):.1f}°C")
        
        with col2:
            st.metric("Humidity", f"{current.get('relative_humidity_2m', 0):.0f}%")
        
        with col3:
            st.metric("Wind Speed", f"{current.get('wind_speed_10m', 0):.1f} m/s")
        
    except Exception as e:
        st.error(f"Error predicting climate zone: {str(e)}")
        st.info("Note: Climate zone prediction requires the exact feature set used during training.")


def anomaly_detection_page():
    """Anomaly detection page"""
    
    st.header("🔍 Anomaly Detection")
    
    models = st.session_state.get('models', {})
    anomaly_detector = models.get('anomaly_detector')
    
    if anomaly_detector is None:
        st.warning("Anomaly Detector model not loaded. Please run notebook 10 first.")
        st.markdown("""
        ### Anomaly Detection Features:
        
        - Detect unusual weather patterns
        - Multiple detection algorithms:
          - Isolation Forest
          - One-Class SVM
          - Local Outlier Factor
          - Elliptic Envelope
        - Ensemble voting for robust detection
        - Anomaly scoring and severity assessment
        
        ### Use Cases:
        - Extreme weather event detection
        - Sensor data validation
        - Climate pattern anomalies
        - Early warning system
        """)
        return
    
    if st.session_state.weather_data is None:
        st.warning("Please fetch weather data from the sidebar first.")
        return
    
    weather = st.session_state.weather_data
    aqi = st.session_state.aqi_data
    
    # Extract features from weather data
    current = weather.get('current', {})
    
    try:
        # Create feature vector for anomaly detection
        # Note: This is a simplified version. In production, you'd need to match
        # the exact features used during training
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Extract relevant features
        features = np.array([[
            current.get('temperature_2m', 0),
            current.get('relative_humidity_2m', 0),
            current.get('wind_speed_10m', 0),
            current.get('pressure_msl', 0),
            current.get('precipitation', 0),
            current.get('uv_index', 0)
        ]])
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Use ensemble detection if available
        if hasattr(anomaly_detector, 'ensemble_detection'):
            # This would require the detector to be re-fitted or have predict method
            # For now, we'll show a placeholder
            st.info("Anomaly detection requires historical data context for comparison.")
            
            # Display current conditions for reference
            st.subheader("Current Weather Conditions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Temperature", f"{current.get('temperature_2m', 0):.1f}°C")
            
            with col2:
                st.metric("Humidity", f"{current.get('relative_humidity_2m', 0):.0f}%")
            
            with col3:
                st.metric("Pressure", f"{current.get('pressure_msl', 0):.0f} hPa")
            
            st.markdown("---")
            st.info("Note: Real-time anomaly detection requires comparing current conditions against historical patterns. The trained model is designed for batch analysis of historical data.")
        else:
            st.warning("Ensemble detection method not available in loaded model.")
        
    except Exception as e:
        st.error(f"Error in anomaly detection: {str(e)}")


def air_quality_page():
    """Air quality page"""
    
    st.header("💨 Air Quality")
    
    if st.session_state.aqi_data is None:
        st.warning("Please fetch weather data from the sidebar first.")
        return
    
    aqi = st.session_state.aqi_data
    
    # Current air quality
    if 'current' in aqi:
        current = aqi['current']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pm25 = current.get('pm2_5', 0)
            st.metric("PM2.5", f"{pm25:.1f} µg/m³")
        
        with col2:
            pm10 = current.get('pm10', 0)
            st.metric("PM10", f"{pm10:.1f} µg/m³")
        
        with col3:
            co = current.get('carbon_monoxide', 0)
            st.metric("CO", f"{co:.1f} µg/m³")
        
        with col4:
            no2 = current.get('nitrogen_dioxide', 0)
            st.metric("NO2", f"{no2:.1f} µg/m³")
        
        # AQI category
        aqi_category = "Good"
        if pm25 > 12:
            aqi_category = "Moderate"
        if pm25 > 35:
            aqi_category = "Unhealthy for Sensitive Groups"
        if pm25 > 55:
            aqi_category = "Unhealthy"
        if pm25 > 150:
            aqi_category = "Very Unhealthy"
        if pm25 > 250:
            aqi_category = "Hazardous"
        
        st.metric("AQI Category", aqi_category)
    
    st.markdown("---")
    
    st.subheader("Air Quality Standards")
    
    st.markdown("""
    | AQI Category | PM2.5 (µg/m³) | Health Impact |
    |--------------|----------------|---------------|
    | Good | 0-12 | Air quality is satisfactory |
    | Moderate | 12.1-35 | Acceptable quality |
    | Unhealthy for Sensitive | 35.1-55 | Sensitive groups may be affected |
    | Unhealthy | 55.1-150 | Everyone may begin to experience effects |
    | Very Unhealthy | 150.1-250 | Health alert: everyone may experience more serious effects |
    | Hazardous | 250+ | Health warning of emergency conditions |
    """)


def weather_classification_page():
    """Weather classification page"""
    
    st.header("🌤️ Weather Classification")
    
    st.info("Weather classification will be available after training ML models.")
    
    st.markdown("""
    ### Weather Classes:
    
    - Sunny
    - Rain
    - Cloudy
    - Mist
    - Fog
    - Thunderstorm
    
    ### Models:
    - Random Forest
    - XGBoost
    - LightGBM
    """)


def recommendations_page():
    """Recommendations page"""
    
    st.header("💡 Weather Recommendations")
    
    if st.session_state.weather_data is None:
        st.warning("Please fetch weather data from the sidebar first.")
        return
    
    weather = st.session_state.weather_data
    aqi = st.session_state.aqi_data
    
    recommendations = []
    
    # Temperature-based recommendations
    if 'current' in weather:
        temp = weather['current'].get('temperature_2m', 25)
        
        if temp > 35:
            recommendations.append("🔥 **Extreme Heat Warning**: Stay hydrated, avoid outdoor activities during peak hours.")
        elif temp > 30:
            recommendations.append("☀️ **Hot Weather**: Wear light clothing, stay hydrated, use sunscreen.")
        elif temp < 10:
            recommendations.append("🧥 **Cold Weather**: Wear warm clothing, layer with thermal wear.")
    
    # Rain-based recommendations
    if 'current' in weather:
        precip = weather['current'].get('precipitation', 0)
        
        if precip > 10:
            recommendations.append("🌧️ **Heavy Rain**: Carry an umbrella, avoid waterlogged areas.")
        elif precip > 0:
            recommendations.append("🌦️ **Light Rain**: Carry an umbrella, be cautious while driving.")
    
    # Wind-based recommendations
    if 'current' in weather:
        wind = weather['current'].get('wind_speed_10m', 0)
        
        if wind > 15:
            recommendations.append("💨 **Strong Winds**: Secure loose objects, avoid outdoor activities.")
    
    # Air quality-based recommendations
    if aqi and 'current' in aqi:
        pm25 = aqi['current'].get('pm2_5', 0)
        
        if pm25 > 55:
            recommendations.append("😷 **Poor Air Quality**: Wear N95 mask, limit outdoor activities.")
        elif pm25 > 35:
            recommendations.append("😐 **Moderate Air Quality**: Sensitive groups should limit outdoor exposure.")
    
    # UV-based recommendations
    if 'current' in weather:
        uv = weather['current'].get('uv_index', 0)
        
        if uv > 8:
            recommendations.append("🧴 **High UV Index**: Wear sunscreen, seek shade, wear protective clothing.")
        elif uv > 5:
            recommendations.append("😎 **Moderate UV Index**: Apply sunscreen, wear sunglasses.")
    
    # Display recommendations
    if recommendations:
        st.subheader("Personalized Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.info("No specific recommendations at this time. Weather conditions are normal.")


if __name__ == "__main__":
    main()
