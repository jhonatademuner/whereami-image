from fastapi import FastAPI
from app.routers import image

app = FastAPI()

# Include routers
app.include_router(image.router, prefix="/api/image", tags=["Image"])

@app.get("/api/health")
def health_check():
    return {"status": "ok"}