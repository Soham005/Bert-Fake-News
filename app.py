# ==========================================
# Financial News Sentiment Analyzer
# Streamlit Cloud Compatible Version
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
st.markdown("Analyze financial news using FinBERT.")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Settings")

company = st.sidebar.text_input("Company Name", "Apple")
num_articles = st.sidebar.slider("Number of Articles", 5, 20, 10)
api_key = st.sidebar.text_input("NewsAPI Key", type="password")

# ----------------------------
# Load Model (Cached)
# ----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )

sentiment_model = load_model()

# ----------------------------
# Fetch News
# ----------------------------
def fetch_news(query, key):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&language=en&sortBy=publishedAt&pageSize=20&apiKey={key}"
    )
    response = requests.get(url)

    if response.status_code != 200:
        return []

    data = response.json()
    return data.get("articles", [])

# ----------------------------
# Analyze Button
# ----------------------------
if st.button("Analyze News"):

    if not api_key:
        st.error("Please enter your NewsAPI key.")
        st.stop()

    with st.spinner("Fetching news..."):
        articles = fetch_news(company, api_key)

    if not articles:
        st.warning("No news found or API limit reached.")
        st.stop()

    df = pd.DataFrame(articles)

    if "title" not in df.columns:
        st.error("Unexpected API response.")
        st.stop()

    df = df[["title"]].head(num_articles)

    with st.spinner("Running FinBERT..."):
        results = sentiment_model(df["title"].tolist())

    df["Sentiment"] = [r["label"] for r in results]
    df["Confidence"] = [round(r["score"], 3) for r in results]

    st.subheader("News Sentiment Results")
    st.dataframe(df)

    # Sentiment Chart
    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # Overall Score
    score_map = {"positive": 1, "neutral": 0, "negative": -1}

    overall_score = (
        df["Sentiment"]
        .str.lower()
        .map(score_map)
        .mean()
    )

    st.subheader("Overall Sentiment Score")
    st.metric("Score (-1 to +1)", round(overall_score, 3))
