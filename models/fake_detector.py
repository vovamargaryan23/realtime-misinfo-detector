class FakeDetector:
    def __init__(self):
        self.fake_indicators = [
            'miracle cure', 'doctors hate', 'big pharma', 'natural cure',
            'government hiding', 'secret cure', 'instant cure', 'guaranteed',
            'amazing discovery', 'breakthrough', 'revolutionary', 'banned',
            'suppressed', 'conspiracy', 'they don\'t want you to know'
        ]

        self.credible_indicators = [
            'clinical trial', 'peer-reviewed', 'study published', 'research shows',
            'according to', 'medical journal', 'fda approved', 'cdc recommends',
            'who guidelines', 'evidence-based', 'randomized controlled'
        ]

    def predict(self, text: str) -> tuple[bool, float]:
        """Predict if medical text is fake"""
        text_lower = text.lower()

        # Count fake indicators
        fake_score = sum(1 for indicator in self.fake_indicators if indicator in text_lower)

        # Count credible indicators
        credible_score = sum(1 for indicator in self.credible_indicators if indicator in text_lower)

        # Additional checks
        has_exaggerated_claims = any(word in text_lower for word in ['100%', 'guaranteed', 'instant', 'immediate'])
        has_fear_mongering = any(word in text_lower for word in ['dangerous', 'deadly', 'kill you', 'poison'])
        lacks_sources = 'study' not in text_lower and 'research' not in text_lower

        # Scoring logic
        total_fake_score = fake_score
        if has_exaggerated_claims:
            total_fake_score += 0.5
        if has_fear_mongering and lacks_sources:
            total_fake_score += 0.5

        # Decision
        if total_fake_score > credible_score and total_fake_score > 0:
            confidence = min(0.95, 0.6 + total_fake_score * 0.15)
            return True, confidence
        elif credible_score > 0:
            confidence = min(0.95, 0.7 + credible_score * 0.1)
            return False, confidence
        else:
            # Neutral case - slight lean towards real for medical content
            return False, 0.6