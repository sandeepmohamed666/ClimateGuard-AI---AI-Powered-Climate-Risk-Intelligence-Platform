from fastapi import APIRouter

router = APIRouter()

@router.get('/forecast')
async def get_forecast():
    return {
        'horizon': '7 days',
        'forecast': [
            {'date': '2026-08-05', 'temperature': 30.8, 'rainfall': 6.3},
            {'date': '2026-08-06', 'temperature': 31.2, 'rainfall': 8.1},
            {'date': '2026-08-07', 'temperature': 32.0, 'rainfall': 10.4},
            {'date': '2026-08-08', 'temperature': 30.5, 'rainfall': 4.7},
            {'date': '2026-08-09', 'temperature': 29.9, 'rainfall': 12.2},
            {'date': '2026-08-10', 'temperature': 30.1, 'rainfall': 7.0},
            {'date': '2026-08-11', 'temperature': 30.4, 'rainfall': 5.8},
        ],
    }
