import pickle

# Load the saved model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to predict sentiment
def predict_sentiment(text):
    text_tfidf = vectorizer.transform([text])  # Convert input text to TF-IDF
    prediction = model.predict(text_tfidf)  # Get prediction
    return "Positive" if prediction[0] == 2 else "Negative"

# Run prediction loop
while True:
    user_input = input("Enter a comment (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_input)
    print(f"Predicted Sentiment: {sentiment}")