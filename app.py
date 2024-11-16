import yfinance as yf
from transformers import pipeline
import pandas as pd
import ta  
import streamlit as st
import sqlite3
from datetime import datetime

# Streamlit app for stock selection
st.title("Real-Time Financial Analysis")

# Dropdown to select a stock
stock_symbol = st.selectbox(
    "Select a Stock Symbol:",
    options=[
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL",  # Alphabet (Google)
        "AMZN",  # Amazon
        "TSLA",  # Tesla
        "META",  # Meta Platforms (Facebook)
        "NFLX",  # Netflix

        "ADBE",  # Adobe
        "ORCL",  # Oracle
        "INTC",  # Intel
        "IBM",   # IBM
        "CRM",   # Salesforce
        "CSCO",  # Cisco
        "SAP"    # SAP SE
    ],
    index=0
)

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"  # Use PyTorch
    )
def fetch_stock_data(symbol):
    """
    Fetches historical stock data for a given symbol using yfinance.
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1mo", interval="1d")
        if data.empty:
            st.error(f"No stock data found for {symbol}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return None

def fetch_news_data(symbol):
    """
    Fetches news headlines for a given stock symbol using yfinance.
    """
    try:
        stock = yf.Ticker(symbol)
        news_data = stock.news[:5]  # Get top 5 latest news
        if not news_data:
            st.warning(f"No news data found for {symbol}")
            return []
        return news_data
    except Exception as e:
        st.error(f"Error fetching news data for {symbol}: {e}")
        return []

def analyze_sentiment(news_headlines):
    """
    Performs sentiment analysis on a list of news headlines.
    """
    if not news_headlines:
        return []
    sentiment_analyzer = load_sentiment_analyzer()
    sentiments = [{"headline": news['title'], 
                   "sentiment": sentiment_analyzer(news['title'])[0]} for news in news_headlines]
    return sentiments

def analyze_trends(stock_data):
    """
    Analyzes market trends using technical indicators.
    """
    if len(stock_data) < 14:
        st.error("Not enough data to calculate technical indicators.")
        return None
    stock_data['SMA'] = ta.trend.sma_indicator(stock_data['Close'], window=14)
    stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'], window=14)
    return stock_data[['Close', 'SMA', 'RSI']]

def generate_trading_signals(trend_data, sentiment_data):
    """
    Generates trading signals based on market trends and sentiment.
    """
    signals = []
    rsi_value = trend_data['RSI'].iloc[-1]
    
    if not sentiment_data:
        signals.append({"action": "HOLD", "reason": "No sentiment data available."})
        return signals
    
    # Aggregate sentiment
    positive_count = sum(1 for sentiment in sentiment_data if sentiment['sentiment']['label'] == 'POSITIVE')
    negative_count = sum(1 for sentiment in sentiment_data if sentiment['sentiment']['label'] == 'NEGATIVE')
    
    if positive_count > negative_count and rsi_value < 70:
        signals.append({"action": "BUY", "reason": "Overall positive sentiment and RSI below threshold."})
    elif negative_count > positive_count and rsi_value > 70:
        signals.append({"action": "SELL", "reason": "Overall negative sentiment and RSI above threshold."})
    else:
        signals.append({"action": "HOLD", "reason": "Neutral or mixed signals."})
    return signals

def review_signals(signals):
    """
    Displays signals in a Streamlit dashboard for manual review.
    """
    st.title("Trading Signals Review")
    signals_df = pd.DataFrame(signals)
    st.table(signals_df)

def log_trades(signals, symbol):
    """
    Logs trading signals to a SQLite database.
    """
    with sqlite3.connect('trading_signals.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS signals (
            timestamp TEXT, 
            symbol TEXT, 
            action TEXT, 
            reason TEXT
        )''')
        for signal in signals:
            cursor.execute("INSERT INTO signals (timestamp, symbol, action, reason) VALUES (?, ?, ?, ?)",
                           (datetime.now(), symbol, signal['action'], signal['reason']))
        conn.commit()

# Run Analysis
if st.button("Run Analysis"):
    # Fetch data
    stock_data = fetch_stock_data(stock_symbol)
    news_data = fetch_news_data(stock_symbol)
    
    if stock_data is not None:
        # Display stock data
        st.subheader(f"{stock_symbol} Stock Data (Last 30 Days)")
        st.dataframe(stock_data)
        
        # Plot trends
        trend_data = analyze_trends(stock_data)
        if trend_data is not None:
            st.subheader("Stock Trends")
            st.line_chart(trend_data)
            
            # Display news and sentiment analysis
            sentiment_data = analyze_sentiment(news_data)
            st.subheader(f"Latest News for {stock_symbol}")
            for item in sentiment_data:
                st.write(f"**{item['headline']}**")
                st.write(f"Sentiment: {item['sentiment']['label']} (Score: {item['sentiment']['score']:.2f})")
                st.markdown("---")
            
            # Generate signals
            signals = generate_trading_signals(trend_data, sentiment_data)
            st.subheader("Trading Signals")
            review_signals(signals)
            
            # Log trades
            log_trades(signals, stock_symbol)
        else:
            st.error("Trend analysis failed due to insufficient data.")
    else:
        st.error("Unable to fetch data for analysis.")
