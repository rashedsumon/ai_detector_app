import streamlit as st
import numpy as np
from model import get_trained_pipeline

# Configure global page properties
st.set_page_config(
    page_title="AI vs Human Text Classifier",
    page_icon="🤖",
    layout="centered"
)

# App interface headers
st.title("🤖 AI vs Human Text Classifier")
st.write("Determine if a block of text is human-authored or synthesized by an LLM model.")

# Initialize the machine learning pipeline cache safely
@st.cache_resource(show_spinner="Downloading Kaggle data and training model components... Please stand by.")
def initialize_model_backend():
    return get_trained_pipeline()

try:
    classifier_pipeline = initialize_model_backend()
    
except Exception as err:
    st.error(f"Failed to compile application background models: {err}")
    st.stop()

st.divider()

# Interactive User input interface area
user_text_input = st.text_area(
    label="Insert Text Block Below:",
    placeholder="Paste your paragraphs or essays here (at least 15-20 words yields best accuracy results)...",
    height=250
)

# Trigger classification evaluations
if st.button("Analyze Content Properties", type="primary"):
    if not user_text_input.strip():
        st.warning("Analysis block cannot run empty. Please insert text first.")
    else:
        with st.spinner("Evaluating linguistic features..."):
            # Format raw string data to structural sequence lists
            input_data = [user_text_input]
            
            # Extract classifications and probability values
            predicted_class = classifier_pipeline.predict(input_data)[0]
            probability_metrics = classifier_pipeline.predict_proba(input_data)[0]
            classes = classifier_pipeline.classes_
            
            # Find index matching the target predicted labels
            class_index = list(classes).index(predicted_class)
            confidence_score = probability_metrics[class_index] * 100

            # Formatting label structures neatly for web UI presentation
            display_label = "AI Generated" if str(predicted_class).upper() in ["AI", "1", "1.0"] else "Human Written"
            
            st.divider()
            st.subheader("Classification Result:")
            
            # Display localized stylized callout cards depending on output context
            if "AI" in display_label:
                st.error(f"Verdict: **{display_label}**")
                st.metric(label="AI Generation Probability Confidence Score", value=f"{confidence_score:.2f}%")
            else:
                st.success(f"Verdict: **{display_label}**")
                st.metric(label="Human Authorship Probability Confidence Score", value=f"{confidence_score:.2f}%")
                
st.caption("Backend built using Scikit-Learn pipelines using properties provided by the Alitaqishah 2026 Kaggle Dataset benchmarks.")