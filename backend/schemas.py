from pydantic import BaseModel
from typing import List

class TextInput(BaseModel):
    text: str

class AnalysisResult(BaseModel):
    is_medical: bool
    medical_confidence: float
    is_fake: bool
    fake_confidence: float
    evidence: str
    sources: List[str]
    processing_time: float
