from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def get_anomalies():
    return {
        'outlier_percentage': 3.8,
        'normal_percentage': 96.2,
        'anomaly_points': [
            {'date': '2026-07-15', 'feature': 'rainfall', 'value': 124.6},
            {'date': '2026-07-30', 'feature': 'temperature', 'value': 43.2},
        ],
    }
