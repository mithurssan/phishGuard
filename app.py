import streamlit as st
import joblib
from backend.preprocess import preprocess

# Loading the model and vectoriser
model = joblib.load("backend/model/spam_classifier.joblib")
vectoriser = joblib.load("backend/model/tfidf_vectoriser.joblib")

# Streamlit UI
st.set_page_config(page_title="Spam Detector", layout="centered", page_icon="💬")

st.title("💬 Spam Message Classifier")
st.write("Paste in any SMS/email message and find out if it's spam!")

text_input = st.text_area("Enter your message here:", height=150)

if st.button("Classify"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_text = preprocess(text_input)
        vect_text = vectoriser.transform([clean_text])
        prediction = model.predict(vect_text)[0]
        label = "❌ This looks like **spam**." if prediction == 1 else "✅ It's probably **not** spam."
        st.markdown(f"### Result: {label}")

