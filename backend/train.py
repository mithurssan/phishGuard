import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from preprocess import preprocess

def main():
    # Importing the CSV
    df = pd.read_csv("./data/spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Preprocessing the text
    df["clean_text"] = df["text"].apply(preprocess)
    
    # Vectorising
    vectoriser = TfidfVectorizer(max_features=5000)
    X = vectoriser.fit_transform(df["clean_text"])
    y = df["label"]
    
    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=123)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    #Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Saving Model and Vectoriser
    joblib.dump(model, "./backend/model/spam_classifier.joblib")
    joblib.dump(vectoriser, "./backend/model/tfidf_vectoriser.joblib" )
    
if __name__== "__main__":
    main()
