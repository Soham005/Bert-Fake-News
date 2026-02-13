# ==========================================
# Financial News Sentiment Analyzer
# Fully Cloud-Compatible Version
# ==========================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Financial News Sentiment", layout="wide")
st.title("📊 Financial News Sentiment Analysis (FinBERT API)")
st.markdown("Live financial news sentiment using hosted FinBERT model.")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Settings")

company = st.sidebar.text_input("Company Name", "Apple")
num_articles = st.sidebar.slider("Number of Articles", 5, 20, 10)
news_api_key = st.sidebar.text_input("NewsAPI Key", type="password")
hf_api_key = st.sidebar.text_input("HuggingFace API Key", type="password")

# ----------------------------
# Fetch News Function
# ----------------------------
def fetch_news(query, api_key):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return []

    data = response.json()
    return data.get("articles", [])


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
        return None, None

    result = response.json()

    # Handle model loading / error response
    if isinstance(result, dict) and "error" in result:
        return None, None

    try:
        scores = result[0]
        best = max(scores, key=lambda x: x["score"])
        return best["label"].lower(), round(best["score"], 3)
    except:
        return None, None


# ----------------------------
# Analyze Button
# ----------------------------
if st.button("🔍 Analyze News"):

    if not news_api_key or not hf_api_key:
        st.error("Please enter both NewsAPI and HuggingFace API keys.")
        st.stop()

    # Fetch News
    with st.spinner("Fetching news..."):
        articles = fetch_news(company, news_api_key)

    if not articles:
        st.warning("No news found or API limit reached.")
        st.stop()

    df = pd.DataFrame(articles)

    if "title" not in df.columns:
        st.error("Unexpected news format.")
        st.stop()

    df = df[["title"]].head(num_articles)

    sentiments = []
    confidences = []

    # Analyze Sentiment
    with st.spinner("Analyzing sentiment via FinBERT API..."):
        for title in df["title"]:
            label, score = analyze_sentiment_hf(title, hf_api_key)

            # If model is loading, wait once and retry
            if label is None:
                time.sleep(3)
                label, score = analyze_sentiment_hf(title, hf_api_key)

            if label is not None:
                sentiments.append(label)
                confidences.append(score)

    # If no valid predictions
    if not sentiments:
        st.error("FinBERT API did not return predictions. Try again in 1 minute.")
        st.stop()

    df = df.iloc[:len(sentiments)]
    df["Sentiment"] = sentiments
    df["Confidence"] = confidences

    # ----------------------------
    # Display Table
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
    score_map = {"positive": 1, "neutral": 0, "negative": -1}

    mapped_scores = df["Sentiment"].map(score_map)

    if mapped_scores.dropna().empty:
        overall_score = 0
    else:
        overall_score = mapped_scores.mean()

    st.subheader("📈 Overall Sentiment Score")
    st.metric("Score (-1 to +1)", round(overall_score, 3))
