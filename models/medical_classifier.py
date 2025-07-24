import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os


class MedicalClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = MultinomialNB()
        self.medical_keywords = [
            'doctor', 'patient', 'hospital', 'medicine', 'treatment', 'diagnosis',
            'symptom', 'disease', 'illness', 'health', 'medical', 'clinical',
            'therapy', 'surgery', 'drug', 'medication', 'prescription', 'vaccine',
            'virus', 'bacteria', 'infection', 'cancer', 'diabetes', 'covid',
            'coronavirus', 'flu', 'fever', 'pain', 'cure', 'heal', 'remedy'
        ]
        self.is_trained = False
        self._load_or_train()

    def _load_or_train(self):
        """Load existing model or train a simple one"""
        if os.path.exists('medical_classifier.pkl'):
            with open('medical_classifier.pkl', 'rb') as f:
                self.vectorizer, self.model = pickle.load(f)
                self.is_trained = True
        else:
            # For demo purposes, use keyword-based classification
            self.is_trained = False

    def predict(self, text: str) -> tuple[bool, float]:
        """Predict if text is medical-related"""
        if self.is_trained:
            # Use trained model
            text_vec = self.vectorizer.transform([text])
            prob = self.model.predict_proba(text_vec)[0]
            is_medical = self.model.predict(text_vec)[0] == 1
            confidence = max(prob)
            return is_medical, confidence
        else:
            # Use keyword-based approach for demo
            return self._keyword_based_prediction(text)

    def _keyword_based_prediction(self, text: str) -> tuple[bool, float]:
        """Simple keyword-based medical classification"""
        text_lower = text.lower()

        # Count medical keywords
        medical_count = sum(1 for keyword in self.medical_keywords if keyword in text_lower)

        # Simple heuristic
        total_words = len(text.split())
        medical_ratio = medical_count / max(total_words, 1)

        # Decision logic
        if medical_count >= 3 or medical_ratio > 0.1:
            confidence = min(0.95, 0.6 + medical_ratio * 2)
            return True, confidence
        elif medical_count >= 1:
            confidence = 0.3 + medical_ratio
            return medical_ratio > 0.05, confidence
        else:
            return False, 0.9