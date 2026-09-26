from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def get_rainfall():
    return {
        'today': 12.8,
        'weekly': [2.3, 5.4, 12.8, 4.1, 0.0, 0.0, 1.2],
        'high_risk_regions': ['Maharashtra', 'Kerala'],
    }
