import streamlit as st
import requests
import json
import time
import pandas as pd

# Page config
st.set_page_config(
    page_title="Medical Fake News Detector",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.result-card {
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin: 1rem 0;
}

.medical-yes { border-left: 5px solid #27ae60; }
.medical-no { border-left: 5px solid #95a5a6; }
.fake-yes { border-left: 5px solid #e74c3c; }
.fake-no { border-left: 5px solid #27ae60; }

.confidence-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-weight: bold;
    color: white;
    font-size: 0.875rem;
}

.conf-high { background-color: #27ae60; }
.conf-medium { background-color: #f39c12; }
.conf-low { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"


def call_api(endpoint: str, data: dict = None):
    """Call the FastAPI backend"""
    try:
        if data:
            response = requests.post(f"{API_BASE_URL}/{endpoint}", json=data, timeout=30)
        else:
            response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the backend is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_confidence_class(confidence: float) -> str:
    """Get CSS class for confidence level"""
    if confidence >= 0.8:
        return "conf-high"
    elif confidence >= 0.6:
        return "conf-medium"
    else:
        return "conf-low"


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Fake News Detector</h1>
        <p>Simplified AI-Powered Medical Misinformation Detection</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar stats
    with st.sidebar:
        st.header("📊 Statistics")

        stats = call_api("stats")
        if stats:
            st.metric("Total Analyses", stats['total_analyses'])
            st.metric("Medical Posts", f"{stats['medical_posts']} ({stats['medical_percentage']}%)")
            st.metric("Fake Posts", f"{stats['fake_posts']} ({stats['fake_percentage']}%)")
            st.metric("Recent (24h)", stats['recent_analyses'])

    # Main interface
    st.header("🔍 Analyze Text")

    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Example: Essential oils can cure cancer naturally without any side effects...",
        height=150
    )

    # Example buttons
    st.subheader("📝 Try These Examples:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🟢 Medical + Real Example"):
            text_input = st.text_area(
                "Example text:",
                value="According to a clinical trial published in the New England Journal of Medicine, the COVID-19 vaccine showed 95% efficacy in preventing severe illness.",
                height=150,
                key="example1"
            )

    with col2:
        if st.button("🔴 Medical + Fake Example"):
            text_input = st.text_area(
                "Example text:",
                value="BREAKING: Doctors hate this one simple trick! Drinking bleach can instantly cure COVID-19 and cancer. Big pharma has been hiding this miracle cure for decades!",
                height=150,
                key="example2"
            )

    # Analysis button
    if st.button("🚀 Analyze Text", type="primary", disabled=not text_input.strip()):
        if text_input.strip():
            with st.spinner("Analyzing text..."):
                result = call_api("analyze", {"text": text_input})

                if result:
                    # Results display
                    st.header("📋 Analysis Results")

                    # Medical classification
                    st.subheader("1️⃣ Medical Classification")

                    medical_class = "medical-yes" if result['is_medical'] else "medical-no"
                    medical_text = "✅ Medical Content" if result['is_medical'] else "❌ Not Medical Content"
                    conf_class = get_confidence_class(result['medical_confidence'])

                    st.markdown(f"""
                    <div class="result-card {medical_class}">
                        <h4>{medical_text}</h4>
                        <p>Confidence: <span class="confidence-badge {conf_class}">{result['medical_confidence']:.1%}</span></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Fake detection (only if medical)
                    if result['is_medical']:
                        st.subheader("2️⃣ Fake News Detection")

                        fake_class = "fake-yes" if result['is_fake'] else "fake-no"
                        fake_text = "🚨 Likely FAKE" if result['is_fake'] else "✅ Appears REAL"
                        fake_conf_class = get_confidence_class(result['fake_confidence'])

                        st.markdown(f"""
                        <div class="result-card {fake_class}">
                            <h4>{fake_text}</h4>
                            <p>Confidence: <span class="confidence-badge {fake_conf_class}">{result['fake_confidence']:.1%}</span></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Evidence
                        if result['evidence']:
                            st.subheader("🔍 Evidence & Sources")
                            st.info(result['evidence'])

                            if result['sources']:
                                st.write("**Sources:**")
                                for source in result['sources']:
                                    st.write(f"• {source}")

                    # Processing info
                    st.subheader("⚡ Processing Info")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                    with col2:
                        st.metric("Text Length", f"{len(text_input)} chars")


if __name__ == "__main__":
    main()