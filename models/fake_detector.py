# medical-detector/models/fake_detector.py

from transformers import pipeline
import torch

class CompactFakeDetector:
    def __init__(self):
        self.fake_classifier = pipeline(
            "text-classification",
            model="martin-ha/toxic-comment-model",
            device=0 if torch.cuda.is_available() else -1
        )

        self.medical_keywords = [
            'doctor', 'patient', 'medicine', 'treatment', 'diagnosis', 'vaccine',
            'virus', 'covid', 'health', 'medical', 'drug', 'cure', 'symptom'
        ]

        self.fake_indicators = [
            'miracle cure', 'doctors hate', 'big pharma', 'secret cure',
            'guaranteed', 'amazing', 'shocking', 'unbelievable'
        ]

    def is_medical(self, text: str):
        text_lower = text.lower()
        medical_count = sum(1 for word in self.medical_keywords if word in text_lower)
        total_words = len(text.split())

        if medical_count >= 2 or medical_count / max(total_words, 1) > 0.1:
            confidence = min(0.95, 0.6 + medical_count * 0.1)
            return True, confidence
        return False, 0.8

    def detect_fake(self, text: str):
        results = {}

        # ML classifier
        try:
            ml_result = self.fake_classifier(text)[0]
            ml_score = ml_result['score'] if ml_result['label'] == 'TOXIC' else 1 - ml_result['score']
            results['ml_toxic_score'] = ml_score
        except:
            ml_score = 0.5

        # Fake indicators
        text_lower = text.lower()
        fake_count = sum(1 for indicator in self.fake_indicators if indicator in text_lower)
        fake_score = min(fake_count / 3.0, 1.0)
        results['fake_indicators'] = fake_count

        # Credible sources
        credible_sources = ['cdc', 'who', 'fda', 'mayo clinic', 'harvard']
        has_credible = any(source in text_lower for source in credible_sources)
        results['has_credible_source'] = has_credible

        # Ensemble
        final_score = (ml_score * 0.6 + fake_score * 0.3 - (0.2 if has_credible else 0))
        is_fake = final_score > 0.5
        confidence = max(abs(final_score - 0.5) * 2, 0.6)
        results['final_score'] = final_score

        return is_fake, confidence, results
