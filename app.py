# =====================================
# Financial News Sentiment Analyzer
# Streamlit Cloud Safe Version
# =====================================

import streamlit as st
import requests
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="FinBERT News Analyzer", layout="wide")
st.title("📊 Financial News Sentiment Analysis (FinBERT)")
st.markdown("Analyze live financial news sentiment using BERT.")

# ----------------------------
# Sidebar Settings
# ----------------------------
st.sidebar.header("Settings")

stock_keyword = st.sidebar.text_input("Enter Company or Stock Name", "Apple")
num_articles = st.sidebar.slider("Number of Articles", 5, 20, 10)
news_api_key = st.sidebar.text_input("Enter Your NewsAPI Key", type="password")

# ----------------------------
# Load FinBERT Model (Cached)
# ----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# ----------------------------
# Sentiment Function
# ----------------------------
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    labels = ["Negative", "Neutral", "Positive"]

    return labels[predicted_class.item()], confidence.item()

# ----------------------------
# Fetch News Function
# ----------------------------
def fetch_news(query, api_key):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"pageSize=20&"
        f"apiKey={api_key}"
    )
    response = requests.get(url)
    data = response.json()
    return data.get("articles", [])

# ----------------------------
# Main Button
# ----------------------------
if st.button("🔍 Analyze News"):

    if not news_api_key:
        st.error("Please enter your NewsAPI key in the sidebar.")
        st.stop()

    with st.spinner("Fetching news..."):
        articles = fetch_news(stock_keyword, news_api_key)

    if not articles:
        st.warning("No news articles found.")
        st.stop()

    df = pd.DataFrame(articles)
    df = df[['title', 'source', 'url']].head(num_articles)

    sentiments = []
    confidences = []

    with st.spinner("Running FinBERT sentiment analysis..."):
        for headline in df['title']:
            sentiment, confidence = analyze_sentiment(headline)
            sentiments.append(sentiment)
            confidences.append(confidence)

    df['Sentiment'] = sentiments
    df['Confidence'] = confidences

    # ----------------------------
    # Show Data
    # ----------------------------
    st.subheader("📰 News Sentiment Results")
    st.dataframe(df[['title', 'Sentiment', 'Confidence']])

    # ----------------------------
    # Sentiment Distribution Chart
    # ----------------------------
    st.subheader("📊 Sentiment Distribution")

    sentiment_counts = df['Sentiment'].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ----------------------------
    # Overall Sentiment Score
    # ----------------------------
    sentiment_score = (
        df['Sentiment'].map({
            "Positive": 1,
            "Neutral": 0,
            "Negative": -1
        }).mean()
    )

    st.subheader("📈 Overall Sentiment Score")
    st.metric("Score (-1 = Negative, +1 = Positive)", round(sentiment_score, 3))
