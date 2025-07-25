from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.tokenize import sent_tokenize

# Download the 'punkt' tokenizer data for NLTK if you haven't already
try:
    nltk.data.find('tokenizers/punkt') # Keep this check for 'punkt' (the general package)
except LookupError:
    print("NLTK 'punkt' tokenizer data not found. Attempting to download 'punkt'...")
    nltk.download('punkt')
    print("'punkt' download complete.")

# --- ADD THIS BLOCK ---
# Also explicitly download 'punkt_tab' as it's often a dependency that gets missed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK 'punkt_tab' tokenizer data not found. Downloading...")
    nltk.download('punkt_tab')
    print("'punkt_tab' download complete.")
# --- END ADDITION ---

class MedicalKeywordExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the MedicalKeywordExtractor with a Sentence Transformer model.
        Args:
            model_name (str): The name of the pre-trained model to use.
        """
        print(f"Loading Sentence Transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")

        # Define a core set of medical keywords. This can be expanded.
        self.medical_keywords = [
            "organ", "vaccine", "inflammation", "prescription",
            "therapy", "treatment", "diagnosis", "symptom", "disease",
            "cancer", "diabetes", "virus", "infection", "medication",
            "antibiotic", "surgeon", "clinic", "hospital", "patient",
            "medical", "health", "clinical", "pharmaceutical", "therapy",
            "condition", "disorder", "pathology", "anatomy", "physiology"
            # Expanded for better coverage
        ]
        self.medical_vectors = self.model.encode(self.medical_keywords, convert_to_tensor=True)
        print(f"Encoded {len(self.medical_keywords)} medical keywords.")

    def extract(self, text: str, similarity_threshold: float = 0.6) -> dict:
        """
        Extracts medical-related words from a given text based on semantic similarity.
        Args:
            text (str): The input text to analyze.
            similarity_threshold (float): The minimum cosine similarity score to consider a word as medical-related.
        Returns:
            dict: A dictionary mapping identified medical words to their highest similarity score.
        """
        if not text:
            return {}

        # Tokenize and encode each word in the text.
        # We split on spaces for simplicity; a more advanced tokenizer could be used.
        words = text.lower().split()
        if not words:
            return {}

        word_vectors = self.model.encode(words, convert_to_tensor=True)

        extracted_keywords = {}
        for i, word in enumerate(words):
            # Calculate cosine similarity between the word's vector and all medical keyword vectors.
            cosine_scores = util.pytorch_cos_sim(word_vectors[i], self.medical_vectors)[0]

            # Find the highest similarity score for the current word.
            max_similarity = torch.max(cosine_scores).item()

            if max_similarity >= similarity_threshold:
                extracted_keywords[word] = max_similarity

        return extracted_keywords

    def predict(self, text: str, similarity_threshold: float = 0.6) -> bool:
        """
        Predicts whether the given text contains any medical-related content.
        This is a simple binary classification based on whether any keywords
        exceed the similarity threshold in the entire text.

        Args:
            text (str): The input text to analyze.
            similarity_threshold (float): The minimum cosine similarity score
                                          to consider a word as medical-related.
        Returns:
            bool: True if medical keywords are identified, False otherwise.
        """
        extracted_terms = self.extract(text, similarity_threshold)
        return bool(extracted_terms)

    def predict_by_sentence(self, text: str, similarity_threshold: float = 0.6,
                            medical_sentence_ratio_threshold: float = 0.3) -> dict:
        """
        Analyzes a text sentence by sentence to predict whether it contains medical content.
        It identifies which sentences are likely medical and provides an overall prediction.

        Args:
            text (str): The input text to analyze.
            similarity_threshold (float): The minimum cosine similarity score for a word
                                          to be considered medical within a sentence.
            medical_sentence_ratio_threshold (float): The minimum proportion of sentences
                                                      that must be identified as medical
                                                      for the overall text to be classified as medical.

        Returns:
            dict: A dictionary containing:
                - 'is_medical_text': bool, True if the text is classified as medical, False otherwise.
                - 'sentence_details': list, a list of dictionaries for each sentence,
                                        including the sentence text, whether it's medical,
                                        and identified keywords.
        """
        if not text:
            return {
                'is_medical_text': False,
                'sentence_details': []
            }

        sentences = sent_tokenize(text)
        sentence_details = []
        medical_sentence_count = 0

        for sentence in sentences:
            extracted_keywords_in_sentence = self.extract(sentence, similarity_threshold)
            is_sentence_medical = bool(extracted_keywords_in_sentence)

            sentence_details.append({
                'sentence': sentence,
                'is_medical': is_sentence_medical,
                'identified_keywords': extracted_keywords_in_sentence
            })

            if is_sentence_medical:
                medical_sentence_count += 1

        total_sentences = len(sentences)
        overall_is_medical = False
        if total_sentences > 0:
            medical_ratio = medical_sentence_count / total_sentences
            overall_is_medical = medical_ratio >= medical_sentence_ratio_threshold

        return {
            'is_medical_text': overall_is_medical,
            'sentence_details': sentence_details
        }


# Example usage:
def main():
    extractor = MedicalKeywordExtractor()

    test_paragraphs = [
        "The patient presented with severe abdominal pain, a common symptom of gastrointestinal disease. Doctors prescribed a new medication for treatment.",
        "Today's weather is sunny with a high of 25 degrees Celsius. I plan to go for a walk in the park and read a book.",
        "Researchers are developing a novel vaccine to combat the latest virus strain. Clinical trials are underway to assess its efficacy and safety profile.",
        "The cat sat on the mat. The dog barked at the mailman. Birds are singing outside."
    ]

    print("\n--- Running Tests with predict_by_sentence ---")
    for i, paragraph in enumerate(test_paragraphs):
        print(f"\n--- Paragraph {i + 1} ---")
        print(f"Original Text: '{paragraph}'")

        analysis_result = extractor.predict_by_sentence(paragraph, medical_sentence_ratio_threshold=0.5)

        print(f"\nOverall Medical Text Prediction: {analysis_result['is_medical_text']}")
        print("Sentence-by-sentence Analysis:")
        for detail in analysis_result['sentence_details']:
            print(f"  Sentence: '{detail['sentence']}'")
            print(f"    Is Medical: {detail['is_medical']}")
            if detail['identified_keywords']:
                print(f"    Identified Keywords: {', '.join(detail['identified_keywords'].keys())}")
            else:
                print("    No medical keywords identified in this sentence.")

        print("-" * 30)


if __name__ == "__main__":
    main()