import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("amazon_vfl_reviews (1).csv")

# Use correct column names from your dataset
df = df[['review', 'rating']]

# Convert rating to numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Drop missing
df = df.dropna()

# Create sentiment column
df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

X = df['review']
y = df['sentiment']

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_tfidf, y)

# Save model
with open("nlp_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("NLP Model & Vectorizer Saved Successfully!")
