from .dashboard import router as dashboard_router
from .weather import router as weather_router
from .forecasting import router as forecasting_router
from .clustering import router as clustering_router
from .rainfall import router as rainfall_router
from .heatwave import router as heatwave_router
from .anomaly import router as anomaly_router
from .risk import router as risk_router
from .shap import router as shap_router
from .upload import router as upload_router
from .maps import router as maps_router
from .alerts import router as alerts_router
from .aqi import router as aqi_router
from .feature_engineering import router as feature_engineering_router
from .model_metrics import router as model_metrics_router

__all__ = [
    'dashboard_router',
    'weather_router',
    'forecasting_router',
    'clustering_router',
    'rainfall_router',
    'heatwave_router',
    'anomaly_router',
    'risk_router',
    'shap_router',
    'upload_router',
    'maps_router',
    'alerts_router',
    'aqi_router',
    'feature_engineering_router',
    'model_metrics_router',
]
