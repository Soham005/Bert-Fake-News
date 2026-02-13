# ==========================================
# Financial News Sentiment Analyzer
# Cloud-Compatible (No Torch Required)
# ==========================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Financial News Sentiment", layout="wide")
st.title("📊 Financial News Sentiment Analysis (FinBERT API)")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Settings")

company = st.sidebar.text_input("Company Name", "Apple")
num_articles = st.sidebar.slider("Number of Articles", 5, 20, 10)
news_api_key = st.sidebar.text_input("NewsAPI Key", type="password")
hf_api_key = st.sidebar.text_input("HuggingFace API Key", type="password")

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
    return response.json().get("articles", [])

# ----------------------------
# HuggingFace Sentiment API
# ----------------------------
def analyze_sentiment_hf(text, hf_key):
    API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {hf_key}"}

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text}
    )

    if response.status_code != 200:
        return "Error", 0.0

    result = response.json()

    # HF returns list of list
    scores = result[0]
    best = max(scores, key=lambda x: x["score"])

    return best["label"], round(best["score"], 3)

# ----------------------------
# Analyze Button
# ----------------------------
if st.button("Analyze News"):

    if not news_api_key or not hf_api_key:
        st.error("Please enter both NewsAPI and HuggingFace API keys.")
        st.stop()

    with st.spinner("Fetching news..."):
        articles = fetch_news(company, news_api_key)

    if not articles:
        st.warning("No news found.")
        st.stop()

    df = pd.DataFrame(articles)

    if "title" not in df.columns:
        st.error("Unexpected news format.")
        st.stop()

    df = df[["title"]].head(num_articles)

    sentiments = []
    confidences = []

    with st.spinner("Analyzing sentiment via FinBERT API..."):
        for title in df["title"]:
            label, score = analyze_sentiment_hf(title, hf_api_key)
            sentiments.append(label)
            confidences.append(score)

    df["Sentiment"] = sentiments
    df["Confidence"] = confidences

    st.subheader("News Sentiment Results")
    st.dataframe(df)

    # Chart
    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # Overall score
    score_map = {"positive": 1, "neutral": 0, "negative": -1}

    overall_score = (
        df["Sentiment"]
        .str.lower()
        .map(score_map)
        .mean()
    )

    st.subheader("Overall Sentiment Score")
    st.metric("Score (-1 to +1)", round(overall_score, 3))
