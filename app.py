import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load model and vectorizer
try:
    model = joblib.load("news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Fake News Classifier")
st.title("📰 Fake News Classifier")

st.write("Enter a news article to predict whether it's **real** or **fake**.")

news_text = st.text_area("News Article", height=200)

if st.button("Predict"):
    if not news_text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(news_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        result = "✅ REAL" if prediction == 1 else "❌ FAKE"
        st.success(f"Prediction: {result}")
