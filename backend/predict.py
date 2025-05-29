import joblib
from preprocess import preprocess

# Load the model and vectoriser
model = joblib.load("./backend/model/spam_classifier.joblib")
vectoriser = joblib.load("./backend/model/tfidf_vectoriser.joblib")

def predict(text):
    clean_text = preprocess(text)
    vect_text = vectoriser.transform([clean_text])
    prediction = model.predict(vect_text)[0]
    return "spam" if prediction == 1 else "ham"

if __name__ == "__main__":
    sample = "Congratulations! You've won a $5000 gift card. Click here to claim for free!"
    result = predict(sample)
    print(f"Prediction: {result.upper()}")
