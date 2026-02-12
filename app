import streamlit as st
import yfinance as yf
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Financial News Sentiment Analyzer", layout="wide")

st.title("📊 Financial News Sentiment Analysis (FinBERT)")
st.write("Analyze real-time financial news sentiment using BERT.")

# Load FinBERT Model (Cache for performance)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# Sidebar
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)", "AAPL")
num_news = st.sidebar.slider("Number of News Articles", 5, 20, 10)

# Sentiment Function
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    labels = ["Negative", "Neutral", "Positive"]
    return labels[predicted_class], confidence.item()

if st.button("Analyze News"):

    ticker = yf.Ticker(stock_symbol)
    news = ticker.news

    if not news:
        st.error("No news found for this stock.")
    else:
        df = pd.DataFrame(news)
        df = df[['title', 'publisher', 'link']]
        df = df.head(num_news)

        sentiments = []
        confidences = []

        with st.spinner("Analyzing sentiment using FinBERT..."):
            for headline in df['title']:
                sentiment, confidence = analyze_sentiment(headline)
                sentiments.append(sentiment)
                confidences.append(confidence)

        df['Sentiment'] = sentiments
        df['Confidence'] = confidences

        st.subheader("📰 News Sentiment Results")
        st.dataframe(df)

        # Sentiment Distribution
        st.subheader("📊 Sentiment Distribution")

        sentiment_counts = df['Sentiment'].value_counts()

        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax)
        plt.xticks(rotation=0)
        st.pyplot(fig)
