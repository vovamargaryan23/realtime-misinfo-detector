# medical-detector/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from models.fake_detector import CompactFakeDetector

app = FastAPI()

# Load model once at startup
fake_detector = CompactFakeDetector()

class NewsRequest(BaseModel):
    text: str

class FakeResponse(BaseModel):
    is_fake: bool
    confidence: float
    details: dict

@app.post("/detect_fake", response_model=FakeResponse)
async def detect_fake_news(req: NewsRequest):
    is_fake, confidence, details = fake_detector.detect_fake(req.text)
    return FakeResponse(is_fake=is_fake, confidence=confidence, details=details)

# Optional: health check
@app.get("/")
def root():
    return {"status": "API is running"}
