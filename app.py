# ==============================
# Financial News Sentiment App
# Single File - Streamlit Safe
# ==============================

import streamlit as st

# Safe Imports (prevents cloud crash)
try:
    import yfinance as yf
except:
    import subprocess
    subprocess.run(["pip", "install", "yfinance"])
    import yfinance as yf

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except:
    import subprocess
    subprocess.run(["pip", "install", "torch", "transformers"])
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Page Setup
# ------------------------------
st.set_page_config(page_title="FinBERT Sentiment Analyzer", layout="wide")
st.title("📊 Financial News Sentiment Analysis (FinBERT)")
st.markdown("Analyze real-time Yahoo Finance news using BERT NLP model.")

# ------------------------------
# Load Model (Cached)
# ------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("Stock Settings")

stock_symbol = st.sidebar.text_input(
    "Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)", 
    "AAPL"
)

num_articles = st.sidebar.slider(
    "Number of Articles", 
    5, 20, 10
)

# ------------------------------
# Sentiment Function
# ------------------------------
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    labels = ["Negative", "Neutral", "Positive"]

    return labels[predicted_class.item()], confidence.item()

# ------------------------------
# Main Button
# ------------------------------
if st.button("🔍 Analyze News"):

    try:
        ticker = yf.Ticker(stock_symbol)
        news = ticker.news

        if not news:
            st.warning("No news found for this stock.")
            st.stop()

        df = pd.DataFrame(news)

        # Safe column handling
        required_cols = [col for col in ['title', 'publisher', 'link'] if col in df.columns]
        df = df[required_cols].head(num_articles)

        sentiments = []
        confidences = []

        with st.spinner("Running FinBERT model..."):
            for headline in df['title']:
                sentiment, confidence = analyze_sentiment(headline)
                sentiments.append(sentiment)
                confidences.append(confidence)

        df['Sentiment'] = sentiments
        df['Confidence'] = confidences

        # ------------------------------
        # Display Results
        # ------------------------------
        st.subheader("📰 News Sentiment Results")
        st.dataframe(df)

        # ------------------------------
        # Sentiment Chart
        # ------------------------------
        st.subheader("📊 Sentiment Distribution")

        sentiment_counts = df['Sentiment'].value_counts()

        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax)
        plt.xticks(rotation=0)
        st.pyplot(fig)

        # ------------------------------
        # Overall Score
        # ------------------------------
        sentiment_score = (
            df['Sentiment'].map({
                "Positive": 1,
                "Neutral": 0,
                "Negative": -1
            }).mean()
        )

        st.subheader("📈 Overall Sentiment Score")
        st.metric("Sentiment Score (-1 to +1)", round(sentiment_score, 3))

    except Exception as e:
        st.error("Something went wrong while fetching or analyzing data.")
        st.exception(e)
