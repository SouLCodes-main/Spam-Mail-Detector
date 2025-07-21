import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib



def load_data():
    url= "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data= pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

def train_model(data):
    x = data['text']
    y = data['label']
    # Configure vectorizer with better parameters for spam detection
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Use both unigrams and bigrams
        min_df=2,  # Minimum document frequency
        max_df=0.95,  # Maximum document frequency
        stop_words='english'  # Remove English stop words
    )
    x = vectorizer.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = MultinomialNB(alpha=0.1)  # Adjust smoothing parameter
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Test some sample messages before saving
    test_cases = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",  # spam
        "Hello, how are you doing today?",  # not spam
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",  # spam
        "I'll see you at the meeting tomorrow."  # not spam
    ]
    print("\nTesting model with sample messages:")
    x_test_cases = vectorizer.transform(test_cases)
    test_predictions = model.predict(x_test_cases)
    for msg, pred in zip(test_cases, test_predictions):
        print(f"Message: {msg[:50]}...")
        print(f"Prediction: {'Spam' if pred == 1 else 'Not Spam'}\n")
    
    print("Saving model and vectorizer...")
    joblib.dump(model, 'spam_detector_model.pkl')
    joblib.dump(vectorizer, 'spam_detector_vectorizer.pkl')
    return accuracy, report


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print("Data loaded. Training model...")
    accuracy, report= train_model(data)
    print("\nTraining complete!")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")