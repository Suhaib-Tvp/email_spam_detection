# streamlit_app.py
import streamlit as st
import pickle
import re

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.split()
    return ' '.join(text)

# Load models and vectorizers
@st.cache_resource
def load_models():
    with open("spam_classifierLR.pkl", "rb") as f:
        model_lr = pickle.load(f)
    with open("spam_classifierNB.pkl", "rb") as f:
        model_nb = pickle.load(f)
    with open("tfidf_vectorizerLR.pkl", "rb") as f:
        vectorizer_lr = pickle.load(f)
    with open("tfidf_vectorizerNB.pkl", "rb") as f:
        vectorizer_nb = pickle.load(f)
    return model_lr, model_nb, vectorizer_lr, vectorizer_nb

model_lr, model_nb, vectorizer_lr, vectorizer_nb = load_models()

# Streamlit app UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message to check if it's spam or not using either Logistic Regression or Naive Bayes.")

user_input = st.text_area("Type your message here:")
model_choice = st.selectbox("Choose a model:", ["Logistic Regression", "Naive Bayes"])

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned_text = preprocess_text(user_input)

        if model_choice == "Logistic Regression":
            vectorized = vectorizer_lr.transform([cleaned_text]).toarray()
            prediction = model_lr.predict(vectorized)[0]
            prob = model_lr.predict_proba(vectorized)[0][prediction]
        else:
            vectorized = vectorizer_nb.transform([cleaned_text]).toarray()
            prediction = model_nb.predict(vectorized)[0]
            prob = model_nb.predict_proba(vectorized)[0][prediction]

        label = "📬 Ham" if prediction == 0 else "🚫 Spam"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {prob:.2f}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
    
