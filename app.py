# ==========================================
# Financial News Sentiment Analyzer
# Stable Cloud Version (No FinBERT)
# ==========================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Financial News Sentiment", layout="wide")
st.title("📊 Financial News Sentiment Analysis")
st.markdown("Live financial news sentiment analysis.")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Settings")

company = st.sidebar.text_input("Company Name", "Apple")
num_articles = st.sidebar.slider("Number of Articles", 5, 20, 10)
news_api_key = st.sidebar.text_input("NewsAPI Key", type="password")

# ----------------------------
# Fetch News
# ----------------------------
def fetch_news(query, api_key):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return []

    return response.json().get("articles", [])

# ----------------------------
# Sentiment Function (TextBlob)
# ----------------------------
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "positive", polarity
    elif polarity < 0:
        return "negative", polarity
    else:
        return "neutral", polarity

# ----------------------------
# Analyze Button
# ----------------------------
if st.button("🔍 Analyze News"):

    if not news_api_key:
        st.error("Please enter your NewsAPI key.")
        st.stop()

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
    scores = []

    for title in df["title"]:
        label, polarity = analyze_sentiment(title)
        sentiments.append(label)
        scores.append(round(polarity, 3))

    df["Sentiment"] = sentiments
    df["Polarity"] = scores

    # ----------------------------
    # Display Results
    # ----------------------------
    st.subheader("📰 News Sentiment Results")
    st.dataframe(df)

    # ----------------------------
    # Sentiment Distribution
    # ----------------------------
    st.subheader("📊 Sentiment Distribution")

    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ----------------------------
    # Overall Score
    # ----------------------------
    overall_score = df["Polarity"].mean()

    st.subheader("📈 Overall Sentiment Score")
    st.metric("Average Polarity (-1 to +1)", round(overall_score, 3))
