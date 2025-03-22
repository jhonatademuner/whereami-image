from fastapi import APIRouter, UploadFile, File
from app.services.image_service import extract_features

router = APIRouter()

@router.post("/process-image")
async def process_image(location: str, file: UploadFile = File(...)):
    content = await file.read()
    histogram, thumbnail = extract_features(content)
    return {
        "location": location,
        "histogram": histogram.tolist()[:10],
        "thumnnail": thumbnail.tolist()[0][:10],
    }


