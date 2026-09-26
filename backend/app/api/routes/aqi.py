from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def get_aqi():
    return {
        'current': 82,
        'category': 'Moderate',
        'dominant_pollutant': 'PM2.5',
        'trend': [78, 80, 82, 84, 83],
    }
