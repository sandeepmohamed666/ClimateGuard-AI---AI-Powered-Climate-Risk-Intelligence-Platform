from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def get_model_metrics():
    return {
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
        'clustering': {
            'silhouette_score': 0.62,
            'davies_bouldin': 1.02,
            'calinski_harabasz': 325.4,
        },
    }
