from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def get_clusters():
    return {
        'clusters': [
            {'label': 'Coastal risk', 'count': 124},
            {'label': 'Urban heat', 'count': 97},
            {'label': 'Stable climate', 'count': 183},
        ],
        'silhouette_score': 0.62,
        'davies_bouldin': 1.02,
        'calinski_harabasz': 325.4,
    }
