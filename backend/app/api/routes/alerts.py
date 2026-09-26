from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def get_alerts():
    return {
        'alerts': [
            {'title': 'Heatwave Warning', 'level': 'High', 'description': 'Temperatures expected to exceed 40°C.'},
            {'title': 'Heavy Rain', 'level': 'Medium', 'description': 'Localized heavy rainfall expected in western India.'},
            {'title': 'Poor AQI', 'level': 'Moderate', 'description': 'Air quality expected to reach unhealthy levels for sensitive groups.'},
        ]
    }
