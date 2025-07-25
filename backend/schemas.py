from pydantic import BaseModel

class TextInput(BaseModel):
    text: str


class AnalysisResult(BaseModel):
    is_medical: bool
    medical_confidence: float
    is_fake: bool
    fake_confidence: float
    evidence: str
    sources: list
    processing_time: float