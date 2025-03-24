from fastapi import APIRouter, HTTPException, UploadFile, File
from app.services.image_service import compare_images

router = APIRouter()

@router.post("/compare-images")
async def compare_images_route(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()

        hist_similarity, cnn_similarity, mse_similarity, final_similarity = compare_images(img1_bytes, img2_bytes)

        return {
            "histogram_similarity": round(float(hist_similarity) * 100, 2),
            "cnn_similarity": round(float(cnn_similarity) * 100, 2),
            "mse_similarity": round(float(mse_similarity) * 100, 2),
            "final_similarity": round(float(final_similarity), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
