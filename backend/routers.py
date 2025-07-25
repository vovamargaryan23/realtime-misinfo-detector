from fastapi import HTTPException, APIRouter
from schemas import TextInput, AnalysisResult
from models.medical_classifier import MedicalClassifier
from models.fake_detector import FakeDetector
from services.wikipedia_service import WikipediaService
from services.pubmed_service import PubMedService
from database.db import Database
from datetime import datetime


api_router = APIRouter(prefix="")


# Initialize services
medical_classifier = MedicalClassifier()
fake_detector = FakeDetector()
wikipedia_service = WikipediaService()
pubmed_service = PubMedService()
db = Database()


@api_router.post("/analyze", response_model=AnalysisResult)
async def analyze_text(input_data: TextInput):
    """Main analysis endpoint"""
    import time
    start_time = time.time()

    try:
        text = input_data.text

        # Step 1: Check if medical
        is_medical, medical_conf = medical_classifier.predict(text)

        if not is_medical:
            return AnalysisResult(
                is_medical=False,
                medical_confidence=medical_conf,
                is_fake=False,
                fake_confidence=0.0,
                evidence="This text is not medical-related.",
                sources=[],
                processing_time=time.time() - start_time
            )

        # Step 2: Check if fake (only if medical)
        is_fake, fake_conf = fake_detector.predict(text)

        # Step 3: Get evidence
        evidence, sources = await get_evidence(text)

        # Step 4: Store result
        db.store_result({
            'text': text,
            'is_medical': is_medical,
            'medical_confidence': medical_conf,
            'is_fake': is_fake,
            'fake_confidence': fake_conf,
            'timestamp': datetime.now().isoformat()
        })

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
        raise HTTPException(status_code=500, detail=str(e))


async def get_evidence(text: str):
    """Get evidence from multiple sources"""
    evidence_parts = []
    sources = []

    # Get Wikipedia evidence
    wiki_evidence = await wikipedia_service.get_evidence(text)
    if wiki_evidence:
        evidence_parts.append(f"Wikipedia: {wiki_evidence}")
        sources.append("Wikipedia")

    # Get PubMed evidence
    pubmed_evidence = await pubmed_service.get_evidence(text)
    if pubmed_evidence:
        evidence_parts.append(f"PubMed: {pubmed_evidence}")
        sources.append("PubMed")

    evidence = " | ".join(evidence_parts) if evidence_parts else "No evidence found."
    return evidence, sources


@api_router.get("/stats")
async def get_stats():
    """Get analysis statistics"""
    return db.get_stats()
