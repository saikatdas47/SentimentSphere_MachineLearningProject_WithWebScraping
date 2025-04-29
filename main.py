from flask import Flask, jsonify, render_template, request
import joblib
from bs4 import BeautifulSoup
import requests
import logging
from functools import lru_cache
import re
import pandas as pd
import csv
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

# Function to preprocess the input text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Remove emojis
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
      # Removal of numbers
    text = re.sub(r'\d+', '', text)
     # Removal of URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Lowercasing
    text = text.lower()
    return text


# Cache sentiment predictions
@lru_cache(maxsize=1000)
def predict_sentiment(review_text):
    try:
        cleaned_review = preprocess_text(review_text)
        review_vector = vectorizer.transform([cleaned_review])
        sentiment = model.predict(review_vector)

        # Fix: If 1 = Negative, 2 = Positive
        sentiment_label = 'Positive 🥰' if sentiment[0] == 2 else 'Negative 😭'
        return sentiment_label
    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}")
        return "Unknown"

# Function to scrape comments
def scrape_comments(base_url, div_name, max_pages):
    try:
        comments = []
        headers = {"User-Agent": "Mozilla/5.0"}

        for page in range(1, max_pages + 1):
            current_url = f"{base_url}&page={page}"  # Adjust this based on website pagination
            logger.info(f"Scraping: {current_url}")

            response = requests.get(current_url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            found_reviews = soup.find_all('div', class_=div_name)

            if not found_reviews:
                logger.warning(f"No comments found on page {page}. Check the class name.")

            for review in found_reviews:
                review_text = review.get_text(strip=True)
                if review_text:
                    comments.append(review_text)

            print(f"Page {page} - Found {len(found_reviews)} reviews")  # Debugging output

        return comments if comments else ["No comments found. Check URL and class name."]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error scraping comments: {e}")
        return ["Error scraping comments: " + str(e)]
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return ["An unexpected error occurred: " + str(e)]


@app.route('/update_sentiment', methods=['POST'])
def update_sentiment():
    data = request.json
    comment = data['comment']
    new_sentiment = data['sentiment']

    # Convert emoji sentiment to numeric value
    sentiment_label = "2" if new_sentiment == 'Positive 🥰' else "1"

    # Check if the file exists and is not empty
    if not os.path.exists('corrected_comments.csv') or os.stat('corrected_comments.csv').st_size == 0:
        # Create a new file with header if it doesn't exist or is empty
        with open('corrected_comments.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(['Sentiment', 'Comment'])  # Write header
        # Initialize existing data as empty for the first entry
        existing_data = []
    else:
        # Read existing comments to determine if we need to add or delete
        existing_comments = pd.read_csv('corrected_comments.csv')
        existing_data = existing_comments.values.tolist()

    if [sentiment_label, comment] in existing_data:
        # If the comment exists, delete it
        existing_data.remove([sentiment_label, comment])
        operation = "deleted"
    else:
        # If the comment does not exist, add it
        existing_data.append([sentiment_label, comment])
        operation = "added"

    # Append the new data without deleting previous data
    with open('corrected_comments.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        if operation == "added":
            writer.writerow([sentiment_label, comment])  # Append new data
        # Do nothing for "deleted" operation, as it's handled earlier

    # Save the updated data back to the CSV (in case of modifications)
    if operation == "deleted":
        with open('corrected_comments.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(['Sentiment', 'Comment'])  # Write header
            writer.writerows(existing_data)  # Rewrite all rows after removal

    return jsonify({"message": f"Sentiment {operation} successfully!"})

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        div_name = request.form.get('div_name', '').strip()
        max_pages = int(request.form.get('max_pages', 1))

        # Scrape comments
        comments = scrape_comments(url, div_name, max_pages) or ["No comments found."]

        # Ensure comments is always a list
        if not isinstance(comments, list):
            comments = ["Error: Comments could not be retrieved."]

        # Classify sentiment
        classified_comments = [(comment, predict_sentiment(comment)) for comment in comments]

        return render_template('index.html', comments=classified_comments, url=url, div_name=div_name)

    return render_template('index.html', comments=[], url='', div_name='')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)