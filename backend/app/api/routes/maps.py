from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def get_map_layers():
    return {
        'layers': ['heatmap', 'rainfall', 'aqi', 'temperature', 'flood'],
        'india_bounds': [6.5, 68.0, 35.5, 97.5],
    }
