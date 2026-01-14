import streamlit as st
import pandas as pd
import pickle
import re
import unicodedata
from khmernltk import word_tokenize
import joblib

# Set page config
st.set_page_config(
    page_title="Khmer Text Classifier",
    page_icon="🇰🇭",
    layout="centered"
)

# Load stopwords
@st.cache_resource
def load_stopwords():
    try:
        stopwords_df = pd.read_csv("stop_word/khmer_stopwords.csv", header=None)
        return set(stopwords_df[0].tolist())
    except FileNotFoundError:
        st.error("Stopwords file not found. Please check the path.")
        return set()
    
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load("model/RandomForest_TFIDF_Smoke.pkl")
        vectorizer = joblib.load("vectorizers/tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model or vectorizer file not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        return None, None

# Initialize stopwords and pattern
stopwords = load_stopwords()
pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoji (emoticons)
    "\U0001F300-\U0001F5FF"  # emoji (symbols & pictographs)
    "\U0001F680-\U0001F6FF"  # emoji (transport)
    "\U0001F1E0-\U0001F1FF"  # emoji (flags)
    ".,!?;:()\"'។"          # punctuation + Khmer full stop
    "]",
    flags=re.UNICODE
)

# Text preprocessing function
def clean_text(text):
    """Clean and preprocess Khmer text"""
    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", str(text))
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove emojis and punctuation
    text = pattern.sub("", text)
    
    # 4. Tokenize Khmer text
    tokens = word_tokenize(text)
    
    # 5. Remove stopwords and empty tokens
    filtered_tokens = [word for word in tokens if word not in stopwords and word.strip() != ""]
    
    # Join back to string
    return " ".join(filtered_tokens)

# Prediction function
def predict_text(text, model, vectorizer):
    """Preprocess text and make prediction"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    prediction_proba = model.predict_proba(text_vectorized)[0]
    
    return prediction, prediction_proba, cleaned_text

# Main app
def main():
    # Title and description
    st.title("🇰🇭 Khmer Text Classifier")
    st.markdown("Enter a Khmer sentence to classify it using our Random Forest model.")
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Input section
    st.subheader("Input Text")
    user_input = st.text_area(
        "Enter Khmer text:",
        height=150,
        placeholder="បញ្ចូលប្រយោគខ្មែរនៅទីនេះ..."
    )
    
    # Predict button
    if st.button("Predict", type="primary"):
        if user_input.strip():
            with st.spinner("Processing..."):
                try:
                    prediction, prediction_proba, cleaned_text = predict_text(
                        user_input, model, vectorizer
                    )
                    
                    # Display results
                    st.success("Prediction Complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Class", prediction)
                    
                    with col2:
                        confidence = max(prediction_proba) * 100
                        st.metric("Confidence", f"{confidence:.2f}%")
                    
                    # Show probabilities for all classes
                    st.subheader("Class Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': model.classes_,
                        'Probability': prediction_proba
                    }).sort_values('Probability', ascending=False)
                    
                    st.dataframe(prob_df, use_container_width=True)
                    
                    # Show preprocessing details
                    with st.expander("View Preprocessing Details"):
                        st.write("**Original Text:**")
                        st.text(user_input)
                        st.write("**Cleaned Text:**")
                        st.text(cleaned_text)
                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some text to classify.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Random Forest & TF-IDF*")

if __name__ == "__main__":
    main()