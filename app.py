# ==========================================
# Financial News Sentiment Analyzer (FinBERT)
# Production-Ready Streamlit Version
# ==========================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="FinBERT Sentiment Analyzer", layout="wide")
st.title("📊 Financial News Sentiment Analysis")
st.markdown("Analyze real-time financial news using FinBERT.")

# ----------------------------
# Sidebar Settings
# ----------------------------
st.sidebar.header("Settings")

company = st.sidebar.text_input("Enter Company Name", "Apple")
num_articles = st.sidebar.slider("Number of Articles", 5, 20, 10)
news_api_key = st.sidebar.text_input("NewsAPI Key", type="password")

# ----------------------------
# Load FinBERT Model (Cached)
# ----------------------------
@st.cache_resource
def load_model():
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )
    return sentiment_pipeline

sentiment_model = load_model()

# ----------------------------
# Fetch News
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

    if response.status_code != 200:
        return None

    data = response.json()
    return data.get("articles", [])

# ----------------------------
# Analyze Button
# ----------------------------
if st.button("🔍 Analyze News"):

    if not news_api_key:
        st.error("Please enter your NewsAPI key.")
        st.stop()

    with st.spinner("Fetching latest news..."):
        articles = fetch_news(company, news_api_key)

    if not articles:
        st.warning("No news found or API limit reached.")
        st.stop()

    df = pd.DataFrame(articles)

    if "title" not in df.columns:
        st.error("Unexpected API response format.")
        st.stop()

    df = df[["title"]].head(num_articles)

    # ----------------------------
    # Sentiment Analysis
    # ----------------------------
    with st.spinner("Running FinBERT sentiment analysis..."):
        results = sentiment_model(df["title"].tolist())

    df["Sentiment"] = [r["label"] for r in results]
    df["Confidence"] = [round(r["score"], 3) for r in results]

    # ----------------------------
    # Display Results
    # ----------------------------
    st.subheader("📰 News Sentiment Results")
    st.dataframe(df)

    # ----------------------------
    # Sentiment Distribution Chart
    # ----------------------------
    st.subheader("📊 Sentiment Distribution")

    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ----------------------------
    # Overall Sentiment Score
    # ----------------------------
    score_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    sentiment_score = (
        df["Sentiment"]
        .str.lower()
        .map(score_map)
        .mean()
    )

    st.subheader("📈 Overall Sentiment Score")
    st.metric("Score (-1 = Negative, +1 = Positive)", round(sentiment_score, 3))
