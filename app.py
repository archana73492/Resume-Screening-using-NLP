import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.title("📄 Resume Screening App")

user_input = st.text_area("Enter Resume Text")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)

    st.success(f"Predicted Role: {prediction[0]}")
