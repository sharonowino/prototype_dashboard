"""
Weather Integration Module
External API integration for weather covariates
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import streamlit as st
from functools import lru_cache

class WeatherService:
    """Weather data service using OpenWeatherMap API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY", "")
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.cache = {}
        self.cache_ttl = 600  # 10 minutes
    
    def is_available(self) -> bool:
        """Check if weather service is configured."""
        return bool(self.api_key)
    
    @lru_cache(maxsize=100)
    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather for coordinates."""
        if not self.is_available():
            return None
        
        # Round to 0.1° for caching (~10km grid)
        lat_r, lon_r = round(lat, 1), round(lon, 1)
        cache_key = (lat_r, lon_r, datetime.now().minute // 10)
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.now() - cached["fetched"] < timedelta(seconds=self.cache_ttl):
                return cached["data"]
        
        try:
            response = requests.get(
                f"{self.base_url}/weather",
                params={
                    "lat": lat_r,
                    "lon": lon_r,
                    "appid": self.api_key,
                    "units": "metric"
                },
                timeout=5
            )
            data = response.json()
            
            result = {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"].get("speed", 0),
                "wind_deg": data["wind"].get("deg", 0),
                "rain_1h": data.get("rain", {}).get("1h", 0.0),
                "snow_1h": data.get("snow", {}).get("1h", 0.0),
                "clouds": data["clouds"]["all"],
                "weather_id": data["weather"][0]["id"],
                "weather_main": data["weather"][0]["main"],
                "weather_desc": data["weather"][0]["description"],
                "visibility_m": data.get("visibility", 10000),
                "fetched": datetime.now()
            }
            
            self.cache[cache_key] = result
            return result
        except Exception as e:
            st.warning(f"Weather API error: {e}")
            return None
    
    def get_forecast(self, lat: float, lon: float, hours: int = 3) -> Optional[Dict]:
        """Get short-term forecast."""
        if not self.is_available():
            return None
        
        try:
            response = requests.get(
                f"{self.base_url}/forecast",
                params={
                    "lat": round(lat, 1),
                    "lon": round(lon, 1),
                    "appid": self.api_key,
                    "units": "metric",
                    "cnt": hours  # Number of 3-hour intervals
                },
                timeout=5
            )
            return response.json()
        except:
            return None
    
    def compute_weather_impact_score(self, weather: Dict) -> float:
        """
        Compute weather impact score (0-1) based on conditions.
        Higher score = worse conditions for transit.
        """
        if not weather:
            return 0.0
        
        score = 0.0
        
        # Precipitation
        precip = weather.get("rain_1h", 0) + weather.get("snow_1h", 0)
        if precip > 5:  # >5mm/hour heavy
            score += 0.4
        elif precip > 0.5:
            score += 0.2
        
        # Visibility
        vis_km = weather.get("visibility_m", 10000) / 1000
        if vis_km < 1:
            score += 0.3
        elif vis_km < 5:
            score += 0.15
        
        # Weather codes (simplified)
        code = weather.get("weather_id", 800)
        # Thunderstorm: 200-299
        if 200 <= code <= 299:
            score += 0.3
        # Drizzle/rain: 500-599
        elif 500 <= code <= 599:
            score += 0.2
        # Snow: 600-699
        elif 600 <= code <= 699:
            score += 0.3
        # Atmosphere (fog, mist): 700-799
        elif 700 <= code <= 799:
            score += 0.2
        
        # Wind
        wind = weather.get("wind_speed", 0)
        if wind > 15:  # Strong wind >15 m/s
            score += 0.1
        
        return min(1.0, score)

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather covariates to DataFrame."""
    df = df.copy()
    
    if 'lat' not in df.columns or 'lon' not in df.columns:
        st.warning("No lat/lon columns for weather enrichment")
        return df
    
    # Initialize weather service
    if 'weather_service' not in st.session_state:
        api_key = os.environ.get("OPENWEATHER_API_KEY", "")
        st.session_state.weather_service = WeatherService(api_key)
    
    weather_svc = st.session_state.weather_service
    
    if not weather_svc.is_available():
        # Add placeholder columns
        for col in ['weather_temperature', 'weather_precipitation', 'weather_visibility_km',
                    'weather_wind_speed', 'weather_impact_score']:
            df[col] = 0.0
        return df
    
    # Cache weather by region grid
    weather_cache = {}
    
    def get_weather_for_point(lat, lon):
        grid_key = (round(float(lat), 1), round(float(lon), 1))
        if grid_key not in weather_cache:
            weather = weather_svc.get_current_weather(lat, lon)
            weather_cache[grid_key] = weather
        return weather_cache[grid_key]
    
    # Add weather columns
    weather_cols = ['temperature', 'precipitation', 'visibility_m', 'wind_speed', 'weather_id', 'impact_score']
    for col in weather_cols:
        df[f'weather_{col}'] = 0.0
    
    for idx, row in df.iterrows():
        try:
            weather = get_weather_for_point(row['lat'], row['lon'])
            if weather:
                df.at[idx, 'weather_temperature'] = weather.get('temperature', 0)
                df.at[idx, 'weather_precipitation'] = weather.get('rain_1h', 0) + weather.get('snow_1h', 0)
                df.at[idx, 'weather_visibility_km'] = weather.get('visibility_m', 10000) / 1000
                df.at[idx, 'weather_wind_speed'] = weather.get('wind_speed', 0)
                df.at[idx, 'weather_id'] = weather.get('weather_id', 800)
                df.at[idx, 'weather_impact_score'] = weather_svc.compute_weather_impact_score(weather)
        except Exception as e:
            pass
    
    return df

def render_weather_panel():
    """Display current weather conditions in dashboard."""
    st.subheader("Weather Impact Assessment")
    
    if 'weather_service' not in st.session_state:
        api_key = os.environ.get("OPENWEATHER_API_KEY", "")
        st.session_state.weather_service = WeatherService(api_key)
    
    svc = st.session_state.weather_service
    
    if not svc.is_available():
        st.info("Weather API not configured. Set OPENWEATHER_API_KEY environment variable.")
        return
    
    # Get weather for region center (Netherlands)
    center_lat, center_lon = 52.0, 5.2
    weather = svc.get_current_weather(center_lat, center_lon)
    
    if weather:
        impact = svc.compute_weather_impact_score(weather)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Temperature", f"{weather['temperature']:.1f}°C")
        with col2:
            st.metric("Wind", f"{weather['wind_speed']:.1f} m/s")
        with col3:
            st.metric("Visibility", f"{weather['visibility_m']/1000:.1f} km")
        with col4:
            impact_color = "#EF4444" if impact > 0.5 else "#F59E0B" if impact > 0.2 else "#22C55E"
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; border: 2px solid {impact_color}; border-radius: 10px;">
                <div style="font-size: 12px; color: #94A3B8;">Impact Score</div>
                <div style="font-size: 24px; font-weight: bold; color: {impact_color};">{impact:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption(f"Conditions: {weather['weather_desc'].title()} | Humidity: {weather['humidity']}%")
    else:
        st.warning("Weather data unavailable")
