import joblib
import os

print("Current working directory:", os.getcwd())

model_path = "D:/python/Machine Learning/Spam-Mail-Detector/spam_detector_model.pkl"
vectorizer_path = "D:/python/Machine Learning/Spam-Mail-Detector/spam_detector_vectorizer.pkl"

print(f"Loading model from: {model_path}")
print(f"File exists? {os.path.exists(model_path)}")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

test_messages = [
    "WINNER!! You have won a £1000 prize!",
    "Hi, can we meet tomorrow?",
    "FREE ENTRY! Win a new car!",
    "Meeting at 3pm in the conference room"
]

print("\nTesting predictions:")
for msg in test_messages:
    text_transformed = vectorizer.transform([msg])
    prediction = model.predict(text_transformed)
    print(f"\nMessage: {msg}")
    print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Not Spam'}")
