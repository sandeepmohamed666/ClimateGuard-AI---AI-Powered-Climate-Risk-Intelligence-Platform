from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def risk_summary():
    return {
        'composite_score': 72,
        'heatwave_risk': 'Medium',
        'flood_risk': 'High',
        'rainfall_risk': 'Medium',
        'wind_risk': 'Low',
        'aqi_risk': 'Moderate',
        'district_ranking': [
            {'district': 'Mumbai', 'rank': 1, 'risk': 'High'},
            {'district': 'Delhi', 'rank': 2, 'risk': 'Medium'},
        ],
        'state_ranking': [
            {'state': 'Maharashtra', 'rank': 1, 'risk': 'High'},
            {'state': 'Rajasthan', 'rank': 2, 'risk': 'Medium'},
        ],
    }
