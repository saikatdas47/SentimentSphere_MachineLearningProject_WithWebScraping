import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load train dataset (no headers in file)
df_train = pd.read_csv("train.csv", header=None)
df_train.columns = ['Label', 'Comments', 'Description']

# Load test dataset (no headers in file)
df_test = pd.read_csv("test.csv", header=None)
df_test.columns = ['Label', 'Comments', 'Description']

# Fill missing values with empty string
df_train.fillna("", inplace=True)
df_test.fillna("", inplace=True)

# Combine comments and descriptions for both train and test sets
X_train = df_train['Comments'] + " " + df_train['Description']
y_train = df_train['Label']
X_test = df_test['Comments'] + " " + df_test['Description']
y_test = df_test['Label']

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")