from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post('')
async def upload_dataset(file: UploadFile = File(...)):
    return {'filename': file.filename, 'status': 'received'}
