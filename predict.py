import joblib
import sys

def load_model():
    model_path = "D:/python/Machine Learning/Spam-Mail-Detector/spam_detector_model.pkl"
    vectorizer_path = "D:/python/Machine Learning/Spam-Mail-Detector/spam_detector_vectorizer.pkl"
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict(text):
    try:
        model, vectorizer = load_model()
        text_transformed = vectorizer.transform([text])
        prediction = model.predict(text_transformed)
        return 'Spam' if prediction[0] == 1 else 'Not Spam'
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Please provide a message to classify")
            sys.exit(1)
        text = sys.argv[1]
        print(f"Analyzing message: {text}")
        prediction = predict(text)
        if prediction:
            print(f'The message is: {prediction}')
    except Exception as e:
        print(f"Error: {str(e)}")