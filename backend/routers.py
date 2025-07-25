import uvicorn
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from schemas import (TextInput, AnalysisResult)
from models.medical_classifier import MedicalClassifier
from models.fake_detector import FakeDetector
from services.wikipedia_service import WikipediaService
from services.pubmed_service import PubMedService
from database.db import Database

# -------------------------------
# Initialize App and Middleware
# -------------------------------
app = FastAPI(title="Medical Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Initialize Services
# -------------------------------
medical_classifier = MedicalClassifier()
fake_detector = FakeDetector()
wikipedia_service = WikipediaService()
pubmed_service = PubMedService()
db = Database()

# -------------------------------
# Define Router
# -------------------------------
api_router = APIRouter()

@api_router.post("/analyze", response_model=AnalysisResult)
async def analyze_text(input_data: TextInput):
    start_time = time.time()
    try:
        print("Received input:", input_data)
        text = input_data.text

        print("Checking if medical...")
        is_medical, medical_conf = medical_classifier.predict(text)
        print(f"is_medical: {is_medical}, confidence: {medical_conf}")

        if not is_medical:
            return AnalysisResult(
                is_medical=is_medical,
                medical_confidence=medical_conf,
                is_fake=False,
                fake_confidence=0.0,
                evidence="Not a medical statement.",
                sources=[],
                processing_time=time.time() - start_time
            )

        print("Checking if fake...")
        is_fake, fake_conf = fake_detector.predict(text)
        print(f"is_fake: {is_fake}, confidence: {fake_conf}")

        print("Getting evidence...")
        evidence, sources = await get_evidence(text)
        print("Evidence:", evidence)

        print("Storing result...")
        db.store_result({
            'text': text,
            'is_medical': is_medical,
            'medical_confidence': medical_conf,
            'is_fake': is_fake,
            'fake_confidence': fake_conf,
            'timestamp': datetime.now().isoformat()
        })

        print("Returning response.")
        return AnalysisResult(
            is_medical=is_medical,
            medical_confidence=medical_conf,
            is_fake=is_fake,
            fake_confidence=fake_conf,
            evidence=evidence,
            sources=sources,
            processing_time=time.time() - start_time
        )


    except Exception as e:
        print("Exception occurred:", e)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/stats")
async def get_stats():
    """Get analysis statistics"""
    return db.get_stats()


# -------------------------------
# Helper Functions
# -------------------------------
async def get_evidence(text: str):
    """Get evidence from multiple sources"""
    evidence_parts = []
    sources = []

    wiki_evidence = await wikipedia_service.get_evidence(text)
    if wiki_evidence:
        evidence_parts.append(f"Wikipedia: {wiki_evidence}")
        sources.append("Wikipedia")

    pubmed_evidence = await pubmed_service.get_evidence(text)
    if pubmed_evidence:
        evidence_parts.append(f"PubMed: {pubmed_evidence}")
        sources.append("PubMed")

    evidence = " | ".join(evidence_parts) if evidence_parts else "No evidence found."
    return evidence, sources


# -------------------------------
# Register Router
# -------------------------------
app.include_router(api_router)


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
