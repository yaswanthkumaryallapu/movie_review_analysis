import streamlit as st
import joblib
import re

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'<br />', ' ', text)  # Remove HTML line breaks
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    return text

# Predict sentiment
def predict_sentiment(review):
    review_processed = preprocess_text(review)
    review_vectorized = vectorizer.transform([review_processed])
    prediction = model.predict(review_vectorized)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return sentiment

# Streamlit UI
st.title("Movie Review Sentiment Analysis")
st.markdown("This app predicts whether a movie review is **Positive** or **Negative**.")

# User Input
user_review = st.text_area("Enter your movie review here:", height=200)

# Analyze Button
if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment = predict_sentiment(user_review)
        if sentiment == "Positive":
            st.success("🎉 The review is Positive!")
        else:
            st.error("😔 The review is Negative!")
