import streamlit as st
from models.fake_detector import CompactFakeDetector

def main():
    st.set_page_config(page_title="🔍 Medical Fake News Detector", layout="wide")

    st.title("🔍 ML-Powered Medical Fake News Detector")
    st.markdown("*Compact version using DistilBERT - Perfect for hackathons!*")

    # Initialize detector
    @st.cache_resource
    def load_detector():
        return CompactFakeDetector()

    detector = load_detector()

    # Input
    user_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Paste medical news or claims here...",
        height=150
    )

    if st.button("🔍 Analyze", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing with ML models..."):
                # Check if medical
                is_medical, med_conf = detector.is_medical(user_input)

                if not is_medical:
                    st.warning("⚠️ This doesn't appear to be medical content")
                    st.metric("Medical Confidence", f"{med_conf:.1%}")
                    return

                # Detect fake news
                is_fake, fake_conf, details = detector.detect_fake(user_input)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Medical Content", "✅ Yes", f"{med_conf:.1%}")

                with col2:
                    if is_fake:
                        st.metric("Fake Detection", "🚨 LIKELY FAKE", f"{fake_conf:.1%}")
                        st.error("⚠️ This content shows signs of misinformation!")
                    else:
                        st.metric("Fake Detection", "✅ Likely Real", f"{fake_conf:.1%}")
                        st.success("✅ This content appears credible")

                # Details
                with st.expander("🔍 Analysis Details"):
                    st.json(details)

                    # Quick evidence
                    if details.get('has_credible_source'):
                        st.info("📚 Contains references to credible sources")
                    if details.get('fake_indicators', 0) > 0:
                        st.warning(f"⚠️ Found {details['fake_indicators']} fake news indicators")
        else:
            st.error("Please enter some text to analyze")

    # Example texts
    with st.expander("📝 Try these examples"):
        examples = [
            "COVID-19 vaccines are safe and effective according to CDC clinical trials",
            "MIRACLE CURE! Doctors HATE this simple trick that cures cancer instantly!",
            "New study published in Nature shows promising results for diabetes treatment"
        ]

        for i, example in enumerate(examples):
            if st.button(f"Example {i + 1}", key=f"ex_{i}"):
                st.text_area("", example, key=f"example_{i}", height=100)
if __name__ == "__main__":
            main()
