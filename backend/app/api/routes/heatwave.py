from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def get_heatwave():
    return {
        'status': 'advisory',
        'regions': ['Rajasthan', 'Gujarat', 'Maharashtra'],
        'temperatures': [38.4, 39.1, 40.2],
    }
