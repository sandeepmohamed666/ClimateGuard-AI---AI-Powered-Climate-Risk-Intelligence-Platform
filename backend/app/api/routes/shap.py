from fastapi import APIRouter

router = APIRouter()

@router.get('')
async def shap_insights():
    return {
        'summary': [{'feature': 'temperature', 'impact': 0.21}, {'feature': 'humidity', 'impact': 0.17}],
        'waterfall': [],
        'force': [],
    }
