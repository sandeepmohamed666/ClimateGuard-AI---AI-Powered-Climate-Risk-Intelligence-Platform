from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def get_dashboard():
    return {
        'weather': {
            'temperature': 29.3,
            'humidity': 65,
            'rainfall': 12.8,
            'pressure': 1008,
            'wind': 8.6,
            'cloud': 42,
            'uv': 6,
            'visibility': 10,
            'aqi': 82,
        },
        'risk': {
            'overall_score': 68,
            'overall': 'Moderate',
            'heatwave': 'Low',
            'flood': 'Medium',
            'rainfall': 'High',
            'air_quality': 'Moderate',
            'storm': 'Low',
        },
        'alerts': [
            {'title': 'Heatwave advisory', 'description': 'High heat expected across western regions.'},
            {'title': 'Heavy rain expected', 'description': 'Localized heavy rainfall in coastal districts.'},
            {'title': 'Elevated AQI', 'description': 'Air quality likely to worsen in urban centers.'},
        ],
        'trends': {
            'temperature': [28.1, 28.7, 29.3, 29.0, 28.5],
            'rainfall': [2.3, 5.4, 12.8, 4.1, 0.0],
        },
        'model_metrics': {
            'rainfall': {
                'accuracy': 0.88,
                'precision': 0.84,
                'recall': 0.81,
                'roc_auc': 0.91,
                'confusion_matrix': [[220, 34], [28, 198]],
            },
            'heatwave': {
                'accuracy': 0.83,
                'precision': 0.79,
                'recall': 0.76,
                'roc_auc': 0.87,
                'f1_score': 0.81,
            },
            'anomaly': {
                'accuracy': 0.90,
                'precision': 0.88,
                'recall': 0.85,
                'support': 520,
            },
            'risk': {
                'mae': 0.14,
                'rmse': 0.22,
                'r2': 0.76,
            },
        },
    }
