from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import (
    dashboard_router,
    weather_router,
    forecasting_router,
    clustering_router,
    rainfall_router,
    heatwave_router,
    anomaly_router,
    risk_router,
    shap_router,
    upload_router,
    maps_router,
    alerts_router,
    aqi_router,
    feature_engineering_router,
    model_metrics_router,
)

app = FastAPI(title='ClimateGuard AI API', version='3.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(dashboard_router, prefix='/api/dashboard', tags=['dashboard'])
app.include_router(weather_router, prefix='/api/weather', tags=['weather'])
app.include_router(forecasting_router, prefix='/api', tags=['forecasting'])
app.include_router(clustering_router, prefix='/api/clusters', tags=['clusters'])
app.include_router(rainfall_router, prefix='/api/rainfall', tags=['rainfall'])
app.include_router(heatwave_router, prefix='/api/heatwave', tags=['heatwave'])
app.include_router(anomaly_router, prefix='/api/anomalies', tags=['anomalies'])
app.include_router(risk_router, prefix='/api/risk', tags=['risk'])
app.include_router(shap_router, prefix='/api/shap', tags=['shap'])
app.include_router(upload_router, prefix='/api/upload', tags=['upload'])
app.include_router(maps_router, prefix='/api/maps', tags=['maps'])
app.include_router(alerts_router, prefix='/api/alerts', tags=['alerts'])
app.include_router(aqi_router, prefix='/api/aqi', tags=['aqi'])
app.include_router(feature_engineering_router, prefix='/api/feature-engineering', tags=['feature-engineering'])
app.include_router(model_metrics_router, prefix='/api/model-metrics', tags=['model-metrics'])

@app.get('/api/health')
async def health_check():
    return {'status': 'ok'}
